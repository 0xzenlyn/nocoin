"""
$NOCOIN mining loop for agent `tututlemot`.

Usage:
    pip install -r requirements.txt
    cp .env.example .env     # or edit inline
    python miner.py

Behavior:
* GET a puzzle for our wallet
* Solve locally via `solver.solve`
* POST the proof
* On WRONG: try variant answers (auto-shrunk LLM output, aliases) then
  ask LLM for a different canonical form before giving up.
* Respect rate limits (max ~8 submissions per 10s; back off on 429)
* Idle + re-poll when the pool is exhausted
* Honors overrides.json: { "<puzzle_id>": "canonical answer" }
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from solver import solve as solver_solve  # noqa: E402
import llm_solver  # noqa: E402

API_URL = "https://bqrapnlqqtjedjyhlfci.supabase.co/functions/v1/submit-solution"

ETH_ADDRESS = os.environ.get("ETH_ADDRESS", "0xA7c5516d130B4393C49B06D2312aE0Efe0463FBe")
AGENT_NAME = os.environ.get("AGENT_NAME", "tututlemot")
SUPABASE_ANON_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJxcmFwbmxxcXRqZWRqeWhsZmNpIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3NzgyNzUyNjQsImV4cCI6MjA5Mzg1MTI2NH0."
    "mf0fz6kAnK0yeAXrb-XT6yikbdRmeAq5jsikVPPhaFE",
)

IDLE_POLL_SECONDS = float(os.environ.get("IDLE_POLL_SECONDS", "60"))
ACTIVE_POLL_SECONDS = float(os.environ.get("ACTIVE_POLL_SECONDS", "1.5"))
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

HTTP_CONNECT_TIMEOUT = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT = float(os.environ.get("HTTP_READ_TIMEOUT", "90"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "10"))

# How many different answers to try per puzzle before giving up.
# variants + one retry LLM call is usually enough.
MAX_ANSWER_ATTEMPTS = int(os.environ.get("MAX_ANSWER_ATTEMPTS", "5"))

MIN_GAP_BETWEEN_SUBMITS = 1.25  # keep <=8 submissions / 10s

OVERRIDES_FILE = Path(__file__).with_name("overrides.json")
LOG_FILE = Path(__file__).with_name("miner.log")

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("nocoin")


def _load_overrides() -> dict[str, str]:
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        log.warning("overrides.json parse error: %s", e)
    return {}


# ---------------------------------------------------------------------------
# HTTP session with exponential backoff
# ---------------------------------------------------------------------------

class Client:
    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update({
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "User-Agent": f"nocoin-miner/{AGENT_NAME}",
        })
        self._last_submit_at = 0.0

    def fetch_puzzle(self) -> Optional[dict]:
        url = f"{API_URL}?eth={ETH_ADDRESS}"
        data = self._request("GET", url)
        if not data:
            return None
        puzzle = data.get("puzzle")
        if puzzle is None:
            log.info("no puzzle available: %s", data.get("message", ""))
            return None
        return puzzle

    def submit(self, puzzle_id: str, answer: str) -> Optional[dict]:
        gap = time.time() - self._last_submit_at
        if gap < MIN_GAP_BETWEEN_SUBMITS:
            time.sleep(MIN_GAP_BETWEEN_SUBMITS - gap)
        payload = {
            "eth_address": ETH_ADDRESS,
            "agent_name": AGENT_NAME,
            "puzzle_id": puzzle_id,
            "answer": answer,
        }
        data = self._request(
            "POST",
            API_URL,
            json_body=payload,
            extra_headers={"Content-Type": "application/json"},
        )
        self._last_submit_at = time.time()
        return data

    def _request(
        self,
        method: str,
        url: str,
        json_body: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> Optional[dict]:
        backoff = 2.0
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.s.request(
                    method,
                    url,
                    json=json_body,
                    headers=extra_headers or {},
                    timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
                )
            except requests.RequestException as e:
                log.warning(
                    "network error (%s) attempt %d/%d: %s",
                    type(e).__name__, attempt + 1, MAX_RETRIES, e,
                )
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                log.warning("429 rate-limited, sleeping %.1fs", retry_after)
                time.sleep(retry_after + random.random())
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code >= 500:
                log.warning("server %s, retry in %.1fs", resp.status_code, backoff)
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60)
                continue

            try:
                return resp.json()
            except ValueError:
                log.warning("non-JSON response %s: %s", resp.status_code, resp.text[:200])
                return None
        log.error("giving up after retries: %s %s", method, url)
        return None


# ---------------------------------------------------------------------------
# Answer-generation: cascade of local solver, overrides, LLM + variants, LLM retry
# ---------------------------------------------------------------------------

def _answer_candidates(puzzle: dict, overrides: dict[str, str]) -> list[str]:
    """Build an ordered list of candidate answers for this puzzle.

    Note: LLM.retry is not included here — it's called dynamically by the
    mining loop after WRONG responses, so it can see which candidates the
    server rejected.
    """
    pid = puzzle.get("id", "")
    prompt = puzzle.get("prompt", "")
    out: list[str] = []

    # 1. Manual override by puzzle id or by normalized prompt.
    if pid in overrides:
        out.append(overrides[pid])
    prompt_key = " ".join(prompt.lower().split())
    if prompt_key in overrides:
        out.append(overrides[prompt_key])

    # 2. Deterministic local solver (hashing/math/trivia/encoding).
    det = solver_solve(puzzle)
    if det:
        out.append(det)

    # 3. Primary LLM attempt.
    if llm_solver.is_enabled():
        llm_ans = llm_solver.solve(puzzle)
        if llm_ans:
            out.append(llm_ans)
            # 4. Shorter/alias variants derived from the LLM answer.
            out.extend(llm_solver.variants_for(llm_ans))

    # Dedup, preserve order, drop empties.
    seen: set[str] = set()
    deduped: list[str] = []
    for a in out:
        a_norm = " ".join(str(a).strip().lower().split())
        if a_norm and a_norm not in seen:
            seen.add(a_norm)
            deduped.append(a_norm)
    return deduped


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_STOP = False


def _handle_sigint(signum, frame):  # noqa: ANN001
    global _STOP
    _STOP = True
    log.info("caught signal %s, shutting down after current puzzle", signum)


def main() -> int:
    if not ETH_ADDRESS.startswith("0x") or len(ETH_ADDRESS) != 42:
        log.error("ETH_ADDRESS looks invalid: %r", ETH_ADDRESS)
        return 2

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    overrides = _load_overrides()
    log.info("agent=%s wallet=%s", AGENT_NAME, ETH_ADDRESS)
    log.info(
        "http timeouts: connect=%.0fs read=%.0fs; retries=%d; max_answer_attempts=%d",
        HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, MAX_RETRIES, MAX_ANSWER_ATTEMPTS,
    )
    log.info("overrides loaded: %d", len(overrides))
    log.info("llm enabled: %s", llm_solver.is_enabled())

    client = Client()
    solved = 0
    given_up_ids: set[str] = set()
    total_earned = 0
    balance: Optional[int] = None

    while not _STOP:
        puzzle = client.fetch_puzzle()
        if puzzle is None:
            log.info("pool empty; sleeping %.0fs", IDLE_POLL_SECONDS)
            _sleep(IDLE_POLL_SECONDS)
            continue

        pid = puzzle.get("id", "?")
        prompt = puzzle.get("prompt", "")
        category = puzzle.get("category", "?")
        diff = puzzle.get("difficulty", "?")
        reward = puzzle.get("reward", 500)

        log.info("puzzle %s [%s d=%s r=%s] %s", pid, category, diff, reward, prompt)

        if pid in given_up_ids:
            # We exhausted all candidates earlier. Re-poll slowly so the
            # server eventually rotates to a different puzzle (or refuses new
            # attempts after you add an override).
            log.warning("gave up earlier on %s; backing off", pid)
            _sleep(max(ACTIVE_POLL_SECONDS * 4, 10.0))
            # Allow manual override to re-enable retry without restart:
            overrides = _load_overrides()
            if pid in overrides or " ".join(prompt.lower().split()) in overrides:
                log.info("override found for %s; retrying", pid)
                given_up_ids.discard(pid)
            continue

        candidates = _answer_candidates(puzzle, overrides)
        if not candidates:
            log.warning("no candidates for %s (logged to unsolved.jsonl)", pid)
            given_up_ids.add(pid)
            _sleep(ACTIVE_POLL_SECONDS)
            continue

        log.info("candidates (%d): %s", len(candidates),
                 [c[:50] + ("..." if len(c) > 50 else "") for c in candidates[:6]])

        tried: list[str] = []
        correct = False

        while candidates and len(tried) < MAX_ANSWER_ATTEMPTS and not _STOP:
            answer = candidates.pop(0)
            if answer in tried:
                continue
            tried.append(answer)

            log.info("attempt %d: %r", len(tried), answer[:120])
            result = client.submit(pid, answer)
            if result is None:
                log.error("submit returned nothing; re-polling to check state")
                # Don't mark as wrong; exit inner loop and re-poll.
                break

            if result.get("correct"):
                solved += 1
                total_earned += int(result.get("reward", reward))
                balance = result.get("balance", balance)
                log.info(
                    "CORRECT puzzle=%s reward=%s balance=%s total_solved=%s attempt=%d",
                    pid, result.get("reward"), balance, solved, len(tried),
                )
                correct = True
                break

            log.warning("WRONG puzzle=%s attempt=%d answer=%r response=%s",
                        pid, len(tried), answer[:80], json.dumps(result)[:200])

            # If we're running out of candidates and have LLM, ask it for a
            # different canonical form given all the strings the server
            # already rejected.
            if (not candidates
                    and llm_solver.is_enabled()
                    and len(tried) < MAX_ANSWER_ATTEMPTS):
                fresh = llm_solver.retry(puzzle, tried)
                if fresh and fresh not in tried:
                    log.info("LLM retry suggested: %r", fresh)
                    candidates.append(fresh)
                    # Also push variants of the retry answer.
                    for v in llm_solver.variants_for(fresh):
                        if v not in tried and v not in candidates:
                            candidates.append(v)

            _sleep(ACTIVE_POLL_SECONDS)

        if not correct:
            log.warning("gave up on %s after %d attempts; tried=%s",
                        pid, len(tried), tried)
            _log_unsolved(puzzle, tried)
            given_up_ids.add(pid)

        _sleep(ACTIVE_POLL_SECONDS)

    log.info("done. solved=%s earned=%s balance=%s", solved, total_earned, balance)
    return 0


def _log_unsolved(puzzle: dict, tried: list[str]) -> None:
    rec = dict(puzzle)
    rec["_tried"] = tried
    try:
        with open(Path(__file__).with_name("unsolved.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _sleep(seconds: float) -> None:
    end = time.time() + seconds
    while not _STOP and time.time() < end:
        time.sleep(min(0.5, end - time.time()))


if __name__ == "__main__":
    raise SystemExit(main())
