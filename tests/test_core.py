"""
Tests for multi-agent-analyst. Run: pytest tests/ -v

Focus areas: mock-mode determinism (the whole point of the mock is
reproducible pipeline verification), verdict schema compliance, the
supervisor's majority-vote logic including the 3-way-tie case, and the
guarantee that live mode fails loudly without a key rather than silently
degrading.
"""

import json

import pytest

from src.agents import SubagentResult, _mock_majority_vote
from src.llm_client import AnthropicLLMClient, MockLLMClient
from src.pipeline import run_full_analysis


def _sub(name, signal, conf=0.5):
    return SubagentResult(
        name=name,
        verdict={"signal": signal, "confidence": conf, "rationale": "r", "key_evidence": []},
        trace=[],
    )


def test_pipeline_deterministic():
    r1 = run_full_analysis("MSFT", mode="mock")
    r2 = run_full_analysis("MSFT", mode="mock")
    for r in (r1, r2):
        r.pop("elapsed_seconds")
    assert json.dumps(r1, default=str) == json.dumps(r2, default=str)


def test_verdict_schema():
    r = run_full_analysis("AAPL", mode="mock")
    for agent in ("technical", "fundamental", "sentiment"):
        v = r[agent]["verdict"]
        assert v["signal"] in {"bullish", "neutral", "bearish"}
        assert 0.0 <= v["confidence"] <= 1.0
        assert v["rationale"].startswith("[MOCK]")  # mock output must be labeled
    t = r["thesis"]["verdict"]
    assert t["overall_signal"] in {"bullish", "neutral", "bearish"}


def test_each_subagent_called_own_tool_only():
    r = run_full_analysis("XOM", mode="mock")
    assert [c["tool_name"] for c in r["technical"]["trace"]] == ["get_technical_snapshot"]
    assert [c["tool_name"] for c in r["fundamental"]["trace"]] == ["get_fundamentals_snapshot"]
    assert [c["tool_name"] for c in r["sentiment"]["trace"]] == ["get_recent_headlines"]
    assert r["thesis"]["trace"] == []  # supervisor has no data tools


def test_majority_vote_clear_majority():
    v = _mock_majority_vote([_sub("a", "bullish"), _sub("b", "bullish"), _sub("c", "bearish")])
    assert v["overall_signal"] == "bullish"
    assert len(v["conflicting_points"]) == 1


def test_majority_vote_three_way_tie_defaults_neutral():
    v = _mock_majority_vote([_sub("a", "bullish"), _sub("b", "neutral"), _sub("c", "bearish")])
    assert v["overall_signal"] == "neutral"


def test_majority_vote_unanimous():
    v = _mock_majority_vote([_sub("a", "bearish"), _sub("b", "bearish"), _sub("c", "bearish")])
    assert v["overall_signal"] == "bearish"
    assert v["conflicting_points"] == []


def test_live_client_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMClient()


def test_mock_ticker_extraction():
    from src.llm_client import _extract_ticker

    assert _extract_ticker("Analyze ticker: NVDA") == "NVDA"
    with pytest.raises(ValueError):
        _extract_ticker("no ticker in this sentence at all")
