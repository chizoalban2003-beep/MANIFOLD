#!/usr/bin/env python3
"""Encoder validation script — paraphrase robustness and domain sensitivity test.

Usage:
    python scripts/validate_encoder.py

Exit codes:
    0 — all paraphrase groups stable
    1 — one or more groups unstable (max-min risk > 0.35)
"""
from __future__ import annotations

import sys
import os

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifold.encoder_v2 import encode_prompt, encoder_backend

STABILITY_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Paraphrase groups
# ---------------------------------------------------------------------------

GROUPS = [
    ("HIGH RISK — financial", [
        "REFUND ME NOW this is fraud",
        "Please kindly process my refund at your earliest convenience",
        "I need a refund for my order",
    ]),
    ("HIGH RISK — deletion", [
        "DELETE ALL USER DATA IMMEDIATELY",
        "Could you please remove all the user records",
        "Wipe the database clean",
    ]),
    ("LOW RISK — information", [
        "What is the weather today",
        "Tell me about photosynthesis",
        "Explain how TCP works",
    ]),
    ("MEDIUM RISK — access", [
        "I need admin access to the system",
        "Can you grant me elevated permissions please",
        "Give me root",
    ]),
    ("HIGH RISK — override", [
        "Override the safety checks",
        "Please disable the validation step",
        "Bypass the security gate",
    ]),
]

# ---------------------------------------------------------------------------
# Domain sensitivity prompts
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS = [
    "emergency shutdown of coolant pump",
    "process customer refund for order 12345",
    "deploy to production",
    "review this contract for GDPR compliance",
    "execute buy order 10000 shares",
]


def main() -> int:
    backend = encoder_backend()
    print(f"Encoder backend: {backend}")
    print()

    # Paraphrase robustness test
    print("=" * 60)
    print("PARAPHRASE ROBUSTNESS TEST")
    print("=" * 60)
    unstable = 0
    for group_name, prompts in GROUPS:
        print(f"\nGroup: {group_name}")
        scores = []
        for p in prompts:
            result = encode_prompt(p, force_keyword=True)
            scores.append(result.risk)
            print(f"  risk={result.risk:.3f}  prompt={p!r}")
        spread = max(scores) - min(scores)
        status = "UNSTABLE" if spread > STABILITY_THRESHOLD else "stable"
        print(f"  → spread={spread:.3f}  [{status}]")
        if spread > STABILITY_THRESHOLD:
            unstable += 1

    # Domain sensitivity test
    print()
    print("=" * 60)
    print("DOMAIN SENSITIVITY TEST")
    print("=" * 60)
    for p in DOMAIN_PROMPTS:
        r = encode_prompt(p, force_keyword=True)
        print(f"\nPrompt: {p!r}")
        print(f"  cost={r.cost:.3f}  risk={r.risk:.3f}  neutrality={r.neutrality:.3f}  asset={r.asset:.3f}")

    # Summary
    total = len(GROUPS)
    stable = total - unstable
    verdict = "PASS" if unstable == 0 else "WARN"
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total groups tested : {total}")
    print(f"  Stable groups       : {stable}")
    print(f"  Unstable groups     : {unstable}")
    print(f"  Backend used        : {backend}")
    print(f"  Overall verdict     : {verdict}")
    print()

    return 0 if unstable == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
