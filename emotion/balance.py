"""OpenRouter credit, sampled before and after a judge run.

Judge stages are the only part that costs money, and the money runs out in
silence: the provider starts answering 402, rows simply never reach the matrix,
and the table then reports a result computed on a subsample. That is how the
shame condition went missing for granite. Asking the balance first is cheaper.

Usage:
    python3 -m emotion.balance                    # current credit
    python3 -m emotion.balance --need 13.5        # is that enough for a run
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import urllib.request

URL = "https://openrouter.ai/api/v1/credits"


def _ssl_context() -> ssl.SSLContext:
    """An SSL context backed by certifi when it is installed.

    The system python on a Mac ships without root certificates, so the default
    context fails with CERTIFICATE_VERIFY_FAILED. The balance guard then spent a
    whole night answering "could not determine" and letting runs through
    unchecked. Judge calls were unaffected, since httpx carries certifi itself.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch() -> dict | None:
    """Credit for the key in the environment. None when the key or the network is absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-or-"):
        return None  # not an OpenRouter key, so there is no balance to read
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as r:
            d = json.load(r)["data"]
    except Exception:
        return None
    bought, spent = float(d["total_credits"]), float(d["total_usage"])
    return {"bought": bought, "spent": spent, "left": bought - spent}


def line(prefix: str = "balance") -> str:
    b = fetch()
    if b is None:
        return f"{prefix}: could not be determined (no OpenRouter key, or no network)"
    return f"{prefix}: ${b['left']:.2f} left (bought ${b['bought']:.2f}, spent ${b['spent']:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter credit.")
    ap.add_argument("--need", type=float, default=None,
                    help="how much the run needs; exits 1 when the credit is short")
    args = ap.parse_args()
    b = fetch()
    print(line())
    if b is None or args.need is None:
        return
    if b["left"] < args.need:
        raise SystemExit(
            f"not enough: {args.need:.2f} needed, ${b['left']:.2f} available. "
            f"Top up the account or narrow the run.")
    print(f"${args.need:.2f} is covered, ${b['left'] - args.need:.2f} would remain")


if __name__ == "__main__":
    main()
