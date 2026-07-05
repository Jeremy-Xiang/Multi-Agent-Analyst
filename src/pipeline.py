"""
pipeline.py — Runs all three subagents, then the supervisor, and returns
one structured, JSON-serializable report with a full audit trail.

The audit trail (which tool was called, with what input, returning what
result, for every agent) is the point, not a debugging afterthought:
"auditable, not black-box" only means something if you can actually open
the report and see exactly which data point each verdict came from.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from .agents import (
    run_fundamental_agent,
    run_sentiment_agent,
    run_supervisor_agent,
    run_technical_agent,
)
from .llm_client import AnthropicLLMClient, LLMClient, MockLLMClient


def get_llm_client(mode: str) -> LLMClient:
    if mode == "live":
        return AnthropicLLMClient()
    if mode == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown mode '{mode}'. Use 'live' or 'mock'.")


def run_full_analysis(ticker: str, mode: str = "mock") -> dict:
    ticker = ticker.upper()
    llm_client = get_llm_client(mode)
    t0 = time.time()

    technical = run_technical_agent(ticker, llm_client)
    fundamental = run_fundamental_agent(ticker, llm_client)
    sentiment = run_sentiment_agent(ticker, llm_client)
    supervisor = run_supervisor_agent(ticker, [technical, fundamental, sentiment], llm_client)

    elapsed = time.time() - t0

    def serialize(result):
        return {
            "verdict": result.verdict,
            "trace": [asdict(t) for t in result.trace],
        }

    return {
        "ticker": ticker,
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "technical": serialize(technical),
        "fundamental": serialize(fundamental),
        "sentiment": serialize(sentiment),
        "thesis": serialize(supervisor),
    }
