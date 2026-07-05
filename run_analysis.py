"""
run_analysis.py — CLI entry point.

    python run_analysis.py --ticker AAPL --mock
    python run_analysis.py --ticker AAPL --live      # requires ANTHROPIC_API_KEY

Prints each subagent's verdict, the supervisor's synthesized thesis, and
the full tool-call audit trail underneath each verdict.
"""

from __future__ import annotations

import argparse
import json

from src.pipeline import run_full_analysis


def print_verdict_block(label: str, block: dict) -> None:
    verdict = block["verdict"]
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    if block["trace"]:
        print("\n  --- audit trail ---")
        for call in block["trace"]:
            print(f"  tool: {call['tool_name']}({call['tool_input']})")
            print(f"  result: {json.dumps(call['tool_result'])[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mock", action="store_true", help="Offline, deterministic, no API key needed (default).")
    mode_group.add_argument("--live", action="store_true", help="Real Anthropic API calls. Requires ANTHROPIC_API_KEY.")
    args = parser.parse_args()

    mode = "live" if args.live else "mock"
    print(f"Running multi-agent analysis for {args.ticker} in '{mode}' mode...")
    if mode == "mock":
        print("(This is rule-based, NOT real analysis — see README before drawing any conclusions from it.)")

    report = run_full_analysis(args.ticker, mode=mode)

    print_verdict_block("TECHNICAL ANALYST", report["technical"])
    print_verdict_block("FUNDAMENTAL ANALYST", report["fundamental"])
    print_verdict_block("SENTIMENT ANALYST", report["sentiment"])
    print_verdict_block("SUPERVISOR — SYNTHESIZED THESIS", report["thesis"])

    print(f"\n{'='*70}")
    print(f"Total time: {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
