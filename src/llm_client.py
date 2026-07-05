"""
llm_client.py — The agent tool-use loop, with two implementations:

- AnthropicLLMClient: the real thing. Loops tool calls against Claude until
  the model calls a dedicated "submit_verdict" tool, which is how this
  project gets reliable structured output instead of parsing free text out
  of a chat response.
- MockLLMClient: no network, no API key, fully deterministic. Calls every
  data tool exactly once, then derives a verdict from simple, explicit
  threshold rules on the returned data. This validates every other part of
  the system (data tools, the loop itself, how the supervisor aggregates
  subagent verdicts) without ever pretending to be real model reasoning —
  every mock verdict is prefixed "[MOCK]" and the rationale states the
  literal rule that fired, on purpose, so it can never be mistaken for
  actual analysis.

Swap which one `pipeline.py` uses via the `--live` / `--mock` CLI flag (see
run_analysis.py) or the `mode` field on the FastAPI request (see app.py).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: dict
    tool_result: dict


@dataclass
class AgentRunResult:
    verdict: dict
    trace: list[ToolCallRecord] = field(default_factory=list)


class LLMClient(ABC):
    @abstractmethod
    def run_agent_until_verdict(
        self,
        system_prompt: str,
        user_message: str,
        data_tools: list[dict],
        verdict_tool: dict,
        tool_dispatch: dict[str, Callable],
        max_turns: int = 6,
    ) -> AgentRunResult:
        ...


class AnthropicLLMClient(LLMClient):
    """
    Real agent loop against the Anthropic API. Requires ANTHROPIC_API_KEY
    in the environment. Model defaults to claude-sonnet-4-6 but can be
    overridden (e.g. for a cheaper/faster model on the subagents and a
    stronger one on the supervisor).
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using AnthropicLLMClient, "
                "or use MockLLMClient for offline testing of the pipeline."
            )
        self.client = anthropic.Anthropic()
        self.model = model

    def run_agent_until_verdict(
        self,
        system_prompt: str,
        user_message: str,
        data_tools: list[dict],
        verdict_tool: dict,
        tool_dispatch: dict[str, Callable],
        max_turns: int = 6,
    ) -> AgentRunResult:
        tools = data_tools + [verdict_tool]
        messages = [{"role": "user", "content": user_message}]
        trace: list[ToolCallRecord] = []

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                # Model responded with plain text instead of calling a tool.
                # Nudge it once rather than silently giving up.
                messages.append(
                    {
                        "role": "user",
                        "content": "Please call a tool to gather data, or call submit_verdict "
                        "if you have enough information to finish.",
                    }
                )
                continue

            tool_results = []
            verdict = None
            for block in tool_use_blocks:
                if block.name == verdict_tool["name"]:
                    verdict = block.input
                    continue
                result = tool_dispatch[block.name](**block.input)
                trace.append(ToolCallRecord(block.name, block.input, result))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                )

            if verdict is not None:
                return AgentRunResult(verdict=verdict, trace=trace)

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Agent did not submit a verdict within {max_turns} turns.")


class MockLLMClient(LLMClient):
    """
    Deterministic, offline stand-in. Calls every data tool exactly once,
    then applies simple, named threshold rules to produce a verdict. This
    is for testing the pipeline's plumbing — NOT a substitute for real
    analysis. Every rationale string says exactly which rule fired.
    """

    def run_agent_until_verdict(
        self,
        system_prompt: str,
        user_message: str,
        data_tools: list[dict],
        verdict_tool: dict,
        tool_dispatch: dict[str, Callable],
        max_turns: int = 6,
    ) -> AgentRunResult:
        trace: list[ToolCallRecord] = []
        collected = {}

        for tool in data_tools:
            name = tool["name"]
            # Every data tool in this project takes a single `ticker` arg —
            # extract it from the user message rather than hardcoding it,
            # so the mock works for whatever ticker the caller passed in.
            ticker = _extract_ticker(user_message)
            result = tool_dispatch[name](ticker=ticker)
            trace.append(ToolCallRecord(name, {"ticker": ticker}, result))
            collected[name] = result

        verdict = _mock_verdict_rules(verdict_tool["name"], collected)
        return AgentRunResult(verdict=verdict, trace=trace)


def _extract_ticker(user_message: str) -> str:
    # Every prompt in this project mentions the ticker as an uppercase token;
    # good enough for a deterministic offline mock, not meant to be robust
    # NLP — the real Anthropic client doesn't need this at all, since the
    # model itself decides what to pass as tool input.
    for token in user_message.replace(":", " ").split():
        if token.isupper() and 1 <= len(token) <= 6 and token.isalpha():
            return token
    raise ValueError(f"MockLLMClient couldn't find a ticker in: {user_message!r}")


def _mock_verdict_rules(verdict_tool_name: str, collected: dict) -> dict:
    """Named, explicit rules — not real analysis, just enough to exercise the pipeline."""
    if "get_technical_snapshot" in collected:
        d = collected["get_technical_snapshot"]
        rsi = d.get("rsi_14") or 50
        mom3 = d.get("trailing_return_3m_pct") or 0
        if mom3 > 5 and rsi < 75:
            signal, rule = "bullish", "3m return > +5% with RSI not yet overbought"
        elif mom3 < -5:
            signal, rule = "bearish", "3m return < -5%"
        else:
            signal, rule = "neutral", "3m return within +/-5%"
        return {
            "signal": signal,
            "confidence": 0.5,
            "rationale": f"[MOCK] Rule fired: {rule}. (3m return={mom3}%, RSI14={rsi})",
            "key_evidence": [f"trailing_return_3m_pct={mom3}", f"rsi_14={rsi}"],
        }

    if "get_fundamentals_snapshot" in collected:
        d = collected["get_fundamentals_snapshot"]
        pe = d.get("trailing_pe") or 25
        growth = d.get("revenue_growth_yoy_pct") or 0
        if pe < 20 and growth > 10:
            signal, rule = "bullish", "P/E < 20 with revenue growth > 10%"
        elif pe > 35 and growth < 5:
            signal, rule = "bearish", "P/E > 35 with revenue growth < 5%"
        else:
            signal, rule = "neutral", "no extreme valuation/growth combination"
        return {
            "signal": signal,
            "confidence": 0.5,
            "rationale": f"[MOCK] Rule fired: {rule}. (P/E={pe}, revenue growth={growth}%)",
            "key_evidence": [f"trailing_pe={pe}", f"revenue_growth_yoy_pct={growth}"],
        }

    if "get_recent_headlines" in collected:
        d = collected["get_recent_headlines"]
        headlines = d.get("headlines", [])
        # Crude keyword count, deliberately simple — this is a mock, not a sentiment model.
        positive_words = ("beats", "record", "raise", "expand", "positively")
        negative_words = ("slide", "scrutiny", "cut", "layoffs", "misses")
        pos = sum(any(w in h["headline"].lower() for w in positive_words) for h in headlines)
        neg = sum(any(w in h["headline"].lower() for w in negative_words) for h in headlines)
        if pos > neg:
            signal, rule = "bullish", f"{pos} positive-keyword headlines vs {neg} negative"
        elif neg > pos:
            signal, rule = "bearish", f"{neg} negative-keyword headlines vs {pos} positive"
        else:
            signal, rule = "neutral", f"tied {pos}-{neg} on keyword count"
        return {
            "signal": signal,
            "confidence": 0.4,
            "rationale": f"[MOCK] Rule fired: {rule}.",
            "key_evidence": [h["headline"] for h in headlines[:3]],
        }

    # Supervisor case: no data tools, just synthesizing subagent verdicts
    # already passed in the user_message. Majority vote on signal.
    return {
        "overall_signal": "neutral",
        "confidence": 0.4,
        "summary": "[MOCK] Supervisor placeholder verdict — see run_supervisor_agent for the "
        "real majority-vote logic used in mock mode.",
        "supporting_points": [],
        "conflicting_points": [],
        "risks": ["This is a MOCK run — no real reasoning was performed."],
    }
