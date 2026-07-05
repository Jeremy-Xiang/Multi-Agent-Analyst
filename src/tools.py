"""
tools.py — Tool schemas (Anthropic tool-use format) and the dispatch table
that maps a tool name to the actual Python function that runs it.

Each subagent gets exactly one data tool plus the shared verdict tool — on
purpose. An agent that can only call one data source can't quietly base a
"technical" verdict on fundamentals data it wasn't supposed to look at,
which makes the later supervisor synthesis step meaningful: three genuinely
independent reads on the same ticker, not three calls to the same blob of
context.
"""

from __future__ import annotations

from .fundamentals import get_fundamentals_snapshot
from .headlines import get_recent_headlines
from .market_data import get_technical_snapshot

TECHNICAL_TOOL = {
    "name": "get_technical_snapshot",
    "description": "Get current price, moving averages, RSI, volatility, and trailing returns for a ticker.",
    "input_schema": {
        "type": "object",
        "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
        "required": ["ticker"],
    },
}

FUNDAMENTALS_TOOL = {
    "name": "get_fundamentals_snapshot",
    "description": "Get valuation and quality ratios (P/E, PEG, margins, growth, debt/equity) for a ticker.",
    "input_schema": {
        "type": "object",
        "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
        "required": ["ticker"],
    },
}

HEADLINES_TOOL = {
    "name": "get_recent_headlines",
    "description": "Get recent news headlines about a ticker, most recent first.",
    "input_schema": {
        "type": "object",
        "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
        "required": ["ticker"],
    },
}

# The forced-structured-output tool every subagent must call to finish.
# Using a tool (rather than asking the model to "respond in JSON" as plain
# text) is the reliable way to get structured output from an agentic loop —
# it's schema-validated by the API, not regex-parsed out of a chat message.
SUBAGENT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your final analysis once you've gathered enough information. This ends your turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            "signal": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "description": "1-3 sentences explaining the verdict."},
            "key_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The specific data points that drove this verdict.",
            },
        },
        "required": ["signal", "confidence", "rationale", "key_evidence"],
    },
}

SUPERVISOR_VERDICT_TOOL = {
    "name": "submit_thesis",
    "description": "Submit the final synthesized investment thesis after reviewing all three subagent verdicts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_signal": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string", "description": "2-4 sentence overall read."},
            "supporting_points": {"type": "array", "items": {"type": "string"}},
            "conflicting_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Where the subagents disagreed, stated plainly.",
            },
            "risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overall_signal", "confidence", "summary", "supporting_points", "conflicting_points", "risks"],
    },
}

TOOL_DISPATCH = {
    "get_technical_snapshot": get_technical_snapshot,
    "get_fundamentals_snapshot": get_fundamentals_snapshot,
    "get_recent_headlines": get_recent_headlines,
}
