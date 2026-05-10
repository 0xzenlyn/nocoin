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
* Respect rate limits (max ~8 submissions per 10s; back off on 429)
* Idle + re-poll when the pool is exhausted
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

from solver import solve  # noqa: E402

API_URL = "https://bqrapnlqqtjedjyhlfci.supabase.co/functions/v1/submit-solution"

ETH_ADDRESS = os.environ.get("ETH_ADDRESS", "0xA7c5516d130B4393C49B06D2312aE0Efe0463FBe")
AGENT_NAME = os.environ.get("AGENT_NAME", "tututlemot")
SUPABASE_ANON_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    # Public anon key exposed by nocoin.live/play.
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJxcmFwbmxxcXRqZWRqeWhsZmNpIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3NzgyNzUyNjQsImV4cCI6MjA5Mzg1MTI2NH0."
    "mf0fz6kAnK0yeAXrb-XT6yikbdRmeAq5jsikVPPhaFE",
)

IDLE_POLL_SECONDS = float(os.environ.get("IDLE_POLL_SECONDS", "60"))
ACTIVE_POLL_SECONDS = float(os.environ.get("ACTIVE_POLL_SECONDS", "1.5"))
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

# Supabase edge functions can cold-start very slowly (30+ s on first hit).
# Let the user override if their network is even worse.
HTTP_CONNECT_TIMEOUT = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT = float(os.environ.get("HTTP_READ_TIMEOUT", "90"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "10"))

MIN_GAP_BETWEEN_SUBMITS = 1.25  # keep <=8 submissions / 10s

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
        # Respect the 8-per-10s cap.
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

    log.info("agent=%s wallet=%s", AGENT_NAME, ETH_ADDRESS)
    log.info(
        "http timeouts: connect=%.0fs read=%.0fs; retries=%d",
        HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, MAX_RETRIES,
    )

    client = Client()
    solved = 0
    failed_ids: set[str] = set()
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

        if pid in failed_ids:
            log.warning("same puzzle %s returned again; skipping to avoid loop", pid)
            _sleep(max(ACTIVE_POLL_SECONDS, 3.0))
            continue

        answer = solve(puzzle)
        if answer is None:
            log.warning("no local solution for %s (logged to unsolved.jsonl)", pid)
            failed_ids.add(pid)
            _sleep(ACTIVE_POLL_SECONDS)
            continue

        log.info("answer -> %r", answer)
        result = client.submit(pid, answer)
        if result is None:
            # Submit timed out or errored. Don't mark wrong — server may have
            # already accepted it. Re-poll; if accepted, next GET returns a
            # different puzzle.
            log.error("submit returned nothing; re-polling to check state")
            _sleep(ACTIVE_POLL_SECONDS)
            continue

        if result.get("correct"):
            solved += 1
            total_earned += int(result.get("reward", reward))
            balance = result.get("balance", balance)
            log.info(
                "CORRECT puzzle=%s reward=%s balance=%s total_solved=%s",
                pid, result.get("reward"), balance, solved,
            )
        else:
            log.warning("WRONG puzzle=%s response=%s", pid, json.dumps(result)[:300])
            failed_ids.add(pid)

        _sleep(ACTIVE_POLL_SECONDS)

    log.info("done. solved=%s earned=%s balance=%s", solved, total_earned, balance)
    return 0


def _sleep(seconds: float) -> None:
    end = time.time() + seconds
    while not _STOP and time.time() < end:
        time.sleep(min(0.5, end - time.time()))


if __name__ == "__main__":
    raise SystemExit(main())
