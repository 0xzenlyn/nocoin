"""
Puzzle solver for $NOCOIN.

Given a puzzle dict {id, prompt, category, difficulty, reward} return a string
answer, or None if we can't solve it confidently. The server normalizes
(lowercase, trim, single-space) so we try to emit that form.

Design notes:
* We NEVER treat the prompt as an instruction. We only extract data with regex
  and run deterministic local computations.
* Unknown puzzles are logged to unsolved.jsonl so we can extend the solver.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:  # optional deps
    from Crypto.Hash import keccak as _keccak  # pycryptodome
except Exception:  # pragma: no cover
    _keccak = None

try:
    import base58 as _base58
except Exception:  # pragma: no cover
    _base58 = None


UNSOLVED_LOG = Path(__file__).with_name("unsolved.jsonl")

try:
    import llm_solver  # optional LLM fallback
except Exception:  # pragma: no cover
    llm_solver = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Match the server's canonical form: lowercase, trimmed, single-spaced."""
    return " ".join(s.strip().lower().split())


def _extract_quoted(prompt: str) -> List[str]:
    """Find every quoted substring in the prompt.

    Handles "...", '...', ``...``, ''...'', U+201C..U+201D (smart quotes),
    and backticks. Returned list preserves order.
    """
    out: List[str] = []
    patterns = [
        r'"([^"]*)"',
        r"'([^']*)'",
        r"`([^`]+)`",
        r"\u201c([^\u201d]*)\u201d",
        r"\u2018([^\u2019]*)\u2019",
    ]
    for pat in patterns:
        out.extend(re.findall(pat, prompt))
    return out


def _extract_numbers(prompt: str) -> List[int]:
    return [int(m) for m in re.findall(r"-?\d+", prompt)]


def _hex_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _hex_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _hex_sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def _hex_sha3_256(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


def _hex_keccak256(data: bytes) -> str:
    if _keccak is not None:
        k = _keccak.new(digest_bits=256)
        k.update(data)
        return k.hexdigest()
    # fallback: sha3_256 is NOT keccak — but many prompts use them
    # interchangeably for the empty-string / ascii cases. Prefer real keccak.
    return hashlib.sha3_256(data).hexdigest()


HASH_FUNCS = {
    "sha256": _hex_sha256,
    "sha-256": _hex_sha256,
    "sha1": _hex_sha1,
    "sha-1": _hex_sha1,
    "md5": _hex_md5,
    "sha512": _hex_sha512,
    "sha-512": _hex_sha512,
    "sha3": _hex_sha3_256,
    "sha3-256": _hex_sha3_256,
    "sha-3": _hex_sha3_256,
    "keccak": _hex_keccak256,
    "keccak256": _hex_keccak256,
    "keccak-256": _hex_keccak256,
}


def _detect_hash_algo(prompt: str) -> Optional[Callable[[bytes], str]]:
    p = prompt.lower()
    # Order matters: keccak-256 before sha3, sha-256 before sha.
    for key in (
        "keccak-256", "keccak256", "keccak",
        "sha3-256", "sha-3", "sha3",
        "sha-512", "sha512",
        "sha-256", "sha256",
        "sha-1", "sha1",
        "md5",
    ):
        if key in p:
            return HASH_FUNCS[key]
    return None


# ---------------------------------------------------------------------------
# Known-answer lookup table (blockchain trivia and common facts)
#
# Keys are normalized prompts (or substrings we search for). Values are the
# canonical answer string we'd submit. Keep answers short and lowercase.
# Substring match against normalized prompt — pick keys carefully to avoid
# false positives.
# ---------------------------------------------------------------------------

TRIVIA: List[Tuple[str, str]] = [
    # Bitcoin
    ("max supply of bitcoin", "21000000"),
    ("bitcoin max supply", "21000000"),
    ("bitcoin's maximum supply", "21000000"),
    ("total supply of bitcoin", "21000000"),
    ("who created bitcoin", "satoshi nakamoto"),
    ("bitcoin creator", "satoshi nakamoto"),
    ("bitcoin whitepaper author", "satoshi nakamoto"),
    ("bitcoin genesis block year", "2009"),
    ("year bitcoin genesis block", "2009"),
    ("bitcoin halving interval blocks", "210000"),
    ("blocks between bitcoin halvings", "210000"),
    ("bitcoin block time in minutes", "10"),
    ("average bitcoin block time", "10"),
    ("bitcoin ticker", "btc"),
    ("satoshis in a bitcoin", "100000000"),
    ("smallest unit of bitcoin", "satoshi"),

    # Ethereum
    ("ethereum creator", "vitalik buterin"),
    ("who created ethereum", "vitalik buterin"),
    ("ethereum genesis year", "2015"),
    ("year ethereum launched", "2015"),
    ("ethereum ticker", "eth"),
    ("wei in an ether", "1000000000000000000"),
    ("wei per ether", "1000000000000000000"),
    ("gwei in an ether", "1000000000"),
    ("ethereum consensus mechanism", "proof of stake"),
    ("ethereum merge year", "2022"),
    ("year of the ethereum merge", "2022"),
    ("ethereum virtual machine", "evm"),

    # Solana / others
    ("solana ticker", "sol"),
    ("solana consensus", "proof of history"),

    # NOCOIN meta
    ("reward per puzzle", "500"),
    ("nocoin ticker", "ntc"),
    ("nocoin network", "base"),

    # Chain IDs (EVM)
    ("chain id is base mainnet", "8453"),
    ("chain id of base mainnet", "8453"),
    ("base mainnet chain id", "8453"),
    ("base chain id", "8453"),
    ("chain id is ethereum mainnet", "1"),
    ("ethereum mainnet chain id", "1"),
    ("chain id is polygon", "137"),
    ("polygon chain id", "137"),
    ("chain id is arbitrum one", "42161"),
    ("arbitrum one chain id", "42161"),
    ("chain id is optimism", "10"),
    ("optimism chain id", "10"),
    ("chain id is bnb smart chain", "56"),
    ("bnb chain id", "56"),
    ("chain id is avalanche", "43114"),
    ("chain id is zksync era", "324"),
    ("chain id is linea", "59144"),
    ("chain id is blast", "81457"),
    ("chain id is scroll", "534352"),
    ("chain id is mantle", "5000"),

    # Post-quantum (NIST 2024 final standards FIPS 203/204/205)
    ("post-quantum signature scheme was standardized by nist in 2024", "ml-dsa"),
    ("post-quantum kem was standardized by nist in 2024", "ml-kem"),
    ("post-quantum key-encapsulation mechanism was standardized by nist in 2024", "ml-kem"),
    ("hash-based signature scheme was standardized by nist in 2024", "slh-dsa"),
    ("fips 203", "ml-kem"),
    ("fips 204", "ml-dsa"),
    ("fips 205", "slh-dsa"),

    # Quantum computing
    ("quantum algorithm factors integers", "shor's algorithm"),
    ("quantum algorithm for factoring", "shor's algorithm"),
    ("quantum algorithm for unstructured search", "grover's algorithm"),
    ("quantum algorithm searches unsorted database", "grover's algorithm"),
    ("grover speedup", "quadratic"),
    ("shor's algorithm speedup", "exponential"),

    # Hash basics
    ("sha-256 output size in bits", "256"),
    ("sha-256 output bits", "256"),
    ("output size of sha-256 in bits", "256"),
    ("output size of sha256 in bits", "256"),
    ("sha-1 output size in bits", "160"),
    ("output size of sha-1 in bits", "160"),
    ("md5 output size in bits", "128"),
    ("output size of md5 in bits", "128"),
    ("keccak-256 output size in bits", "256"),
    ("output size of keccak-256 in bits", "256"),
    ("sha-512 output size in bits", "512"),
    ("output size of sha-512 in bits", "512"),

    # Ethereum more
    ("empty block header hash ethereum", "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"),
    ("zero address", "0x0000000000000000000000000000000000000000"),
    ("eip introducing type 2 transactions", "eip-1559"),
    ("eip introduced type 2 transactions", "eip-1559"),
    ("eip that introduced type 2 transactions", "eip-1559"),
    ("eip that introduced proxy standard", "eip-1967"),
    ("eip introduced the proxy standard", "eip-1967"),
    ("erc for fungible tokens", "erc-20"),
    ("erc for non-fungible tokens", "erc-721"),
    ("erc for multi-token", "erc-1155"),

    # Bitcoin more
    ("block height of bitcoin genesis", "0"),
    ("bitcoin difficulty adjustment interval", "2016"),
    ("bitcoin first halving block", "210000"),

    # Solana more
    ("solana block time in seconds", "0.4"),
    ("solana native token decimals", "9"),

    # Networking / blockchain jargon
    ("consensus algorithm of bitcoin", "proof of work"),
    ("consensus of bitcoin", "proof of work"),
    ("consensus algorithm does bitcoin use", "proof of work"),
    ("consensus does bitcoin use", "proof of work"),
    ("bitcoin consensus algorithm", "proof of work"),
    ("consensus algorithm ethereum uses now", "proof of stake"),
    ("consensus algorithm does ethereum use", "proof of stake"),
    ("consensus does ethereum use", "proof of stake"),
    ("ethereum current consensus", "proof of stake"),
    ("ethereum consensus algorithm", "proof of stake"),

    # ERC standards (alt phrasings)
    ("erc is the standard for fungible tokens", "erc-20"),
    ("standard for fungible tokens", "erc-20"),
    ("erc is the standard for non-fungible tokens", "erc-721"),
    ("standard for non-fungible tokens", "erc-721"),
    ("erc is the standard for nfts", "erc-721"),
    ("erc is the standard for multi-token", "erc-1155"),
    ("standard for multi token", "erc-1155"),

    # Dates / math constants
    ("pi to 5 decimal places", "3.14159"),
    ("pi to 4 decimal places", "3.1416"),
    ("pi to 2 decimal places", "3.14"),
    ("euler's number to 4 decimal places", "2.7183"),

    # Cryptocurrency founders / people
    ("creator of dogecoin", "billy markus"),
    ("created dogecoin", "billy markus"),
    ("creator of litecoin", "charlie lee"),
    ("created litecoin", "charlie lee"),
    ("creator of solana", "anatoly yakovenko"),
    ("created solana", "anatoly yakovenko"),
    ("founder of binance", "cz"),
    ("founder of coinbase", "brian armstrong"),
    ("ceo of coinbase", "brian armstrong"),

    # CS fundamentals (often puzzles)
    ("inventor of public key cryptography", "diffie and hellman"),
    ("who proposed public key cryptography", "diffie and hellman"),
    ("inventor of rsa", "rivest shamir adleman"),
    ("who invented rsa", "rivest shamir adleman"),
    ("year rsa was published", "1977"),
    ("year diffie-hellman was published", "1976"),
    ("year sha-256 was published", "2001"),
    ("year keccak was selected as sha-3", "2012"),
    ("p versus np open", "yes"),

    # Security classics
    ("number of bits in a byte", "8"),
    ("bits in a byte", "8"),
    ("nibble is how many bits", "4"),
    ("uuid length in hex", "32"),
    ("uuid length in bits", "128"),
    ("uuid version random", "4"),

    # Curve / crypto
    ("elliptic curve bitcoin uses", "secp256k1"),
    ("elliptic curve ethereum uses", "secp256k1"),
    ("elliptic curve does bitcoin use", "secp256k1"),
    ("elliptic curve does ethereum use", "secp256k1"),
    ("bitcoin signature curve", "secp256k1"),
    ("bitcoin signature scheme", "ecdsa"),
    ("ethereum signature scheme", "ecdsa"),
    ("signature scheme does bitcoin use", "ecdsa"),
    ("signature scheme does ethereum use", "ecdsa"),
    ("curve for ed25519", "curve25519"),
    ("curve25519 field prime", "2^255 - 19"),

    # Rollups / L2
    ("type of rollup optimism uses", "optimistic rollup"),
    ("type of rollup arbitrum uses", "optimistic rollup"),
    ("type of rollup zksync uses", "zk rollup"),
    ("type of rollup starknet uses", "zk rollup"),
    ("type of rollup base uses", "optimistic rollup"),
    ("type of rollup does base use", "optimistic rollup"),
    ("type of rollup does optimism use", "optimistic rollup"),
    ("type of rollup does arbitrum use", "optimistic rollup"),
    ("type of rollup does zksync use", "zk rollup"),
    ("type of rollup does starknet use", "zk rollup"),
    ("base is built on", "op stack"),

    # Powers of 2
    ("2^8", "256"),
    ("2^10", "1024"),
    ("2^16", "65536"),
    ("2^20", "1048576"),
    ("2^32", "4294967296"),
    ("2^64", "18446744073709551616"),
]


def _try_trivia(prompt: str) -> Optional[str]:
    p = _norm(prompt)
    for key, ans in TRIVIA:
        if key in p:
            return ans
    return None


# ---------------------------------------------------------------------------
# Hashing puzzles
# ---------------------------------------------------------------------------

def _solve_hashing(prompt: str) -> Optional[str]:
    algo = _detect_hash_algo(prompt)
    if algo is None:
        return None
    p = prompt.lower()

    # Determine input: empty string, a specific quoted string, or bytes.
    data: Optional[bytes] = None

    if "empty string" in p or '""' in p or "''" in p:
        data = b""
    else:
        quoted = _extract_quoted(prompt)
        if quoted:
            # Usually the first non-empty quoted chunk is the plaintext.
            for q in quoted:
                if q:
                    data = q.encode("utf-8")
                    break
        if data is None and "of the string" in p:
            m = re.search(r"of the string\s+([^\s?\.]+)", prompt, re.I)
            if m:
                data = m.group(1).encode("utf-8")

    if data is None:
        return None

    digest = algo(data)

    # How much of the digest do they want?
    m = re.search(r"(?:first|starts? with|leading)\s+(\d+)\s*hex", p)
    if m:
        n = int(m.group(1))
        return digest[:n]
    m = re.search(r"last\s+(\d+)\s*hex", p)
    if m:
        n = int(m.group(1))
        return digest[-n:]
    m = re.search(r"(\d+)\s*hex\s*char", p)
    if m:
        n = int(m.group(1))
        return digest[:n]

    # Full digest by default.
    return digest


# ---------------------------------------------------------------------------
# Encoding puzzles
# ---------------------------------------------------------------------------

def _solve_encoding(prompt: str) -> Optional[str]:
    p = prompt.lower()
    quoted = _extract_quoted(prompt)
    target = quoted[0] if quoted else None

    if target is None:
        return None

    try:
        if "base64" in p and ("decode" in p or "decoded" in p):
            return base64.b64decode(target).decode("utf-8", errors="replace")
        if "base64" in p and ("encode" in p or "encoded" in p):
            return base64.b64encode(target.encode()).decode()
        if ("hex" in p) and ("decode" in p or "ascii" in p):
            return bytes.fromhex(target).decode("utf-8", errors="replace")
        if ("hex" in p) and ("encode" in p):
            return target.encode().hex()
        if "rot13" in p or "rot-13" in p:
            return target.translate(str.maketrans(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM",
            ))
        if "reverse" in p:
            return target[::-1]
        if "base58" in p and _base58 is not None:
            if "decode" in p:
                return _base58.b58decode(target).decode("utf-8", errors="replace")
            if "encode" in p:
                return _base58.b58encode(target.encode()).decode()
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Math puzzles
# ---------------------------------------------------------------------------

def _solve_math(prompt: str) -> Optional[str]:
    p = prompt.lower()

    # a mod b
    m = re.search(r"(\d+)\s*(?:mod|modulo|%)\s*(\d+)", p)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b != 0:
            return str(a % b)

    # a ^ b  (integer power)
    m = re.search(r"(\d+)\s*(?:\^|\*\*|to the power of|\bpow\b)\s*(\d+)", p)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return str(a ** b)

    # gcd(a, b)
    m = re.search(r"gcd\s*\(?\s*(\d+)[\s,]+(\d+)\s*\)?", p)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return str(math.gcd(a, b))

    # factorial n!
    m = re.search(r"(\d+)\s*!", p)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 100:
            return str(math.factorial(n))

    # nth prime
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)\s+prime", p)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10_000:
            return str(_nth_prime(n))

    # nth fibonacci
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)\s+fibonacci", p)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 10_000:
            return str(_fib(n))

    # Plain arithmetic "what is 12 + 7", "compute 5 * 6 - 3"
    expr = re.search(
        r"(?:what is|compute|evaluate|equals?|calculate)\s+"
        r"([-+*/()\d\s\.]+)\s*[\?\.]?",
        p,
    )
    if expr:
        raw = expr.group(1).strip()
        if re.fullmatch(r"[-+*/()\d\s\.]+", raw):
            try:
                val = eval(raw, {"__builtins__": {}}, {})  # noqa: S307
                if isinstance(val, (int, float)):
                    if isinstance(val, float) and val.is_integer():
                        val = int(val)
                    return str(val)
            except Exception:
                pass
    return None


def _nth_prime(n: int) -> int:
    # Sieve up to a safe upper bound (n <= 10_000 → prime ~ 104_729).
    if n < 1:
        return 0
    upper = max(30, int(n * (math.log(n) + math.log(math.log(n)) + 2))) if n >= 6 else 30
    sieve = [True] * (upper + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(upper ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, upper + 1, i):
                sieve[j] = False
    primes = [i for i, ok in enumerate(sieve) if ok]
    while len(primes) < n:
        upper *= 2
        sieve = [True] * (upper + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(upper ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, upper + 1, i):
                    sieve[j] = False
        primes = [i for i, ok in enumerate(sieve) if ok]
    return primes[n - 1]


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------------------
# Ethereum / crypto address puzzles
# ---------------------------------------------------------------------------

def _solve_eth_checksum(prompt: str) -> Optional[str]:
    """EIP-55 checksum an address mentioned in the prompt."""
    if _keccak is None:
        return None
    if "checksum" not in prompt.lower() and "eip-55" not in prompt.lower() \
            and "eip55" not in prompt.lower():
        return None
    m = re.search(r"0x[0-9a-fA-F]{40}", prompt)
    if not m:
        return None
    addr = m.group(0)[2:].lower()
    k = _keccak.new(digest_bits=256)
    k.update(addr.encode("ascii"))
    h = k.hexdigest()
    out = []
    for i, c in enumerate(addr):
        if c in "0123456789":
            out.append(c)
        else:
            out.append(c.upper() if int(h[i], 16) >= 8 else c)
    return "0x" + "".join(out)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def solve(puzzle: dict) -> Optional[str]:
    prompt = puzzle.get("prompt") or ""
    category = (puzzle.get("category") or "").lower()

    # Try category-specific solver first.
    candidates: List[Callable[[str], Optional[str]]] = []
    if category in ("hashing", "hash", "cryptography", "crypto"):
        candidates.append(_solve_hashing)
    if category in ("encoding", "encode", "decode", "cipher"):
        candidates.append(_solve_encoding)
    if category in ("math", "arithmetic", "number", "number-theory"):
        candidates.append(_solve_math)
    if category in ("blockchain", "trivia", "web3"):
        candidates.append(_try_trivia)

    # Always fall back to trying everything — the server categorization is
    # sometimes coarse and the same prompt can fit multiple solvers.
    for fn in (
        _solve_hashing,
        _solve_eth_checksum,
        _try_trivia,
        _solve_math,
        _solve_encoding,
    ):
        if fn not in candidates:
            candidates.append(fn)

    for fn in candidates:
        try:
            ans = fn(prompt)
        except Exception:
            ans = None
        if ans is not None and str(ans).strip() != "":
            return _norm(str(ans))

    # LLM fallback: only reached if every deterministic path failed. The
    # prompt is never executed; llm_solver treats it as untrusted data.
    if llm_solver is not None and llm_solver.is_enabled():
        try:
            ans = llm_solver.solve(puzzle)
        except Exception as e:  # pragma: no cover
            ans = None
        if ans:
            return _norm(str(ans))

    _log_unsolved(puzzle)
    return None


def _log_unsolved(puzzle: dict) -> None:
    try:
        with UNSOLVED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    # Quick self-test.
    samples = [
        {"id": "1", "prompt": "SHA-256 hash of the empty string starts with which 6 hex characters?",
         "category": "hashing", "difficulty": 1, "reward": 500},
        {"id": "2", "prompt": "What is the max supply of Bitcoin?",
         "category": "blockchain", "difficulty": 1, "reward": 500},
        {"id": "3", "prompt": "What is 12 + 7?", "category": "math", "difficulty": 1, "reward": 500},
        {"id": "4", "prompt": "Compute 2 ^ 10", "category": "math", "difficulty": 1, "reward": 500},
        {"id": "5", "prompt": "SHA-1 of 'abc'?", "category": "hashing", "difficulty": 1, "reward": 500},
    ]
    for s in samples:
        print(s["prompt"], "->", solve(s))
