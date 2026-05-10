"""
Quick balance / progress checker for $NOCOIN.

Usage:
    python balance.py
    python balance.py --verbose     # show next unsolved puzzle too
    python balance.py --eth 0x...   # check any wallet, not just ours

The NOCOIN API doesn't expose a dedicated balance endpoint, but the miner
log tells us every `balance=<n>` the server confirmed. We read that first
(fast), and fall back to poking the submit endpoint with a dummy answer to
trigger the server's error payload (which usually echoes balance back).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_URL = "https://bqrapnlqqtjedjyhlfci.supabase.co/functions/v1/submit-solution"
APIKEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJxcmFwbmxxcXRqZWRqeWhsZmNpIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3NzgyNzUyNjQsImV4cCI6MjA5Mzg1MTI2NH0."
    "mf0fz6kAnK0yeAXrb-XT6yikbdRmeAq5jsikVPPhaFE",
)
DEFAULT_ETH = os.environ.get("ETH_ADDRESS", "0xA7c5516d130B4393C49B06D2312aE0Efe0463FBe")

LOG_FILE = Path(__file__).with_name("miner.log")


def from_miner_log() -> tuple[int | None, int]:
    """Return (last_balance, solved_count) from miner.log, (None, 0) if absent."""
    if not LOG_FILE.exists():
        return None, 0
    last_balance: int | None = None
    solved = 0
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "CORRECT" in line:
                solved += 1
                m = re.search(r"balance=(\d+)", line)
                if m:
                    last_balance = int(m.group(1))
    return last_balance, solved


def from_api(eth: str) -> dict:
    """GET next puzzle; returns {'puzzle': {...} or None, 'message': ...}."""
    headers = {"apikey": APIKEY, "Authorization": f"Bearer {APIKEY}"}
    resp = requests.get(
        f"{API_URL}?eth={eth}",
        headers=headers,
        timeout=(15, 90),
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth", default=DEFAULT_ETH, help="wallet address")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    eth = args.eth
    print(f"wallet: {eth}")

    last_bal, solved = from_miner_log()
    if last_bal is not None:
        print(f"miner.log  -> solved={solved}, last confirmed balance={last_bal} $NTC")
    else:
        print("miner.log  -> no successful submissions recorded yet")

    print()
    print("checking server...")
    try:
        data = from_api(eth)
    except Exception as e:
        print(f"  network error: {e}")
        return 1

    puzzle = data.get("puzzle")
    if puzzle is None:
        print(f"  status: ALL PUZZLES SOLVED - {data.get('message', '')}")
        if last_bal is not None:
            print(f"  estimated final balance: {last_bal} $NTC")
    else:
        print(f"  status: still have puzzles to solve")
        print(f"  next puzzle category={puzzle.get('category')}  "
              f"difficulty={puzzle.get('difficulty')}  "
              f"reward={puzzle.get('reward')} $NTC")
        if args.verbose:
            print(f"  prompt: {puzzle.get('prompt')}")

    # Also attempt a dummy submit to see if server echoes balance in error.
    if args.verbose and puzzle:
        try:
            resp = requests.post(
                API_URL,
                json={
                    "eth_address": eth,
                    "agent_name": "balance-check",
                    "puzzle_id": puzzle["id"],
                    "answer": "__PROBE__",
                },
                headers={
                    "apikey": APIKEY,
                    "Authorization": f"Bearer {APIKEY}",
                    "Content-Type": "application/json",
                },
                timeout=(15, 90),
            )
            print(f"\nprobe response (HTTP {resp.status_code}):")
            print(f"  {resp.text[:400]}")
        except Exception as e:
            print(f"probe failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
