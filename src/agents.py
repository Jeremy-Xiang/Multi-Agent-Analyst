"""
agents.py — The three subagents and the supervisor that synthesizes them.

Each subagent gets a narrow system prompt, exactly one data tool, and the
shared verdict tool. The supervisor gets no data tools at all — it only
ever sees the three subagent verdicts and has to synthesize them, which is
the whole point: it can't quietly go re-derive its own technical read, it
has to actually engage with what the subagents already said.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .llm_client import AgentRunResult, LLMClient, MockLLMClient, ToolCallRecord
from .tools import (
    FUNDAMENTALS_TOOL,
    HEADLINES_TOOL,
    SUBAGENT_VERDICT_TOOL,
    SUPERVISOR_VERDICT_TOOL,
    TECHNICAL_TOOL,
    TOOL_DISPATCH,
)

TECHNICAL_SYSTEM_PROMPT = """\
You are a technical analyst. You only consider price action, momentum, and \
volatility — never headlines or fundamentals, even if you happen to know them. \
Call get_technical_snapshot for the ticker you're given, then call submit_verdict \
with your read. Be willing to say "neutral" — manufacturing a strong opinion from \
weak or mixed signals is worse than admitting the data doesn't point clearly either way."""

FUNDAMENTALS_SYSTEM_PROMPT = """\
You are a fundamental analyst. You only consider valuation and quality metrics — \
P/E, PEG, margins, growth, debt levels — never price action or headlines. \
Call get_fundamentals_snapshot for the ticker you're given, then call submit_verdict \
with your read. Be willing to say "neutral" — manufacturing a strong opinion from \
weak or mixed signals is worse than admitting the data doesn't point clearly either way."""

SENTIMENT_SYSTEM_PROMPT = """\
You are a sentiment analyst. You only consider recent news headlines — never price \
action or fundamentals. Call get_recent_headlines for the ticker you're given, then \
call submit_verdict with your read. Note explicitly if the headline sample is too \
thin or too neutral to support a confident read — don't force a signal that isn't there."""

SUPERVISOR_SYSTEM_PROMPT = """\
You are the supervising analyst. You do not have access to any data tools — your \
job is to synthesize the three subagent verdicts you're given (technical, \
fundamental, sentiment) into one overall thesis. Name where they agree, name \
where they genuinely conflict (don't paper over disagreement), and name at least \
one real risk to your own conclusion. Call submit_thesis when done."""


@dataclass
class SubagentResult:
    name: str
    verdict: dict
    trace: list[ToolCallRecord]


def run_technical_agent(ticker: str, llm_client: LLMClient) -> SubagentResult:
    result = llm_client.run_agent_until_verdict(
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        user_message=f"Analyze ticker: {ticker}",
        data_tools=[TECHNICAL_TOOL],
        verdict_tool=SUBAGENT_VERDICT_TOOL,
        tool_dispatch=TOOL_DISPATCH,
    )
    return SubagentResult("technical", result.verdict, result.trace)


def run_fundamental_agent(ticker: str, llm_client: LLMClient) -> SubagentResult:
    result = llm_client.run_agent_until_verdict(
        system_prompt=FUNDAMENTALS_SYSTEM_PROMPT,
        user_message=f"Analyze ticker: {ticker}",
        data_tools=[FUNDAMENTALS_TOOL],
        verdict_tool=SUBAGENT_VERDICT_TOOL,
        tool_dispatch=TOOL_DISPATCH,
    )
    return SubagentResult("fundamental", result.verdict, result.trace)


def run_sentiment_agent(ticker: str, llm_client: LLMClient) -> SubagentResult:
    result = llm_client.run_agent_until_verdict(
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        user_message=f"Analyze ticker: {ticker}",
        data_tools=[HEADLINES_TOOL],
        verdict_tool=SUBAGENT_VERDICT_TOOL,
        tool_dispatch=TOOL_DISPATCH,
    )
    return SubagentResult("sentiment", result.verdict, result.trace)


def run_supervisor_agent(
    ticker: str, subagent_results: list[SubagentResult], llm_client: LLMClient
) -> SubagentResult:
    if isinstance(llm_client, MockLLMClient):
        # The generic mock (see llm_client.py) calls data tools and applies
        # threshold rules — neither applies to the supervisor, which has no
        # data tools and exists purely to synthesize. So: a real, explicit
        # majority-vote rule instead of routing through the generic mock
        # fallback (which would otherwise just hand back an uninformative
        # static placeholder regardless of what the subagents actually said).
        verdict = _mock_majority_vote(subagent_results)
        return SubagentResult("supervisor", verdict, trace=[])

    subagent_summary = "\n\n".join(
        f"{r.name.upper()} AGENT VERDICT:\n"
        f"  signal: {r.verdict['signal']}\n"
        f"  confidence: {r.verdict['confidence']}\n"
        f"  rationale: {r.verdict['rationale']}\n"
        f"  key_evidence: {r.verdict['key_evidence']}"
        for r in subagent_results
    )
    user_message = f"Ticker: {ticker}\n\n{subagent_summary}\n\nSynthesize these into one overall thesis."

    result = llm_client.run_agent_until_verdict(
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        user_message=user_message,
        data_tools=[],
        verdict_tool=SUPERVISOR_VERDICT_TOOL,
        tool_dispatch={},
    )
    return SubagentResult("supervisor", result.verdict, result.trace)


def _mock_majority_vote(subagent_results: list[SubagentResult]) -> dict:
    signals = [r.verdict["signal"] for r in subagent_results]
    counts = {s: signals.count(s) for s in set(signals)}
    overall = max(counts, key=counts.get)
    is_unanimous = len(set(signals)) == 1
    is_tied = len(signals) == 3 and len(set(signals)) == 3  # one of each, no majority

    if is_tied:
        overall = "neutral"

    supporting = [f"{r.name}: {r.verdict['signal']} ({r.verdict['rationale']})" for r in subagent_results if r.verdict["signal"] == overall]
    conflicting = [f"{r.name}: {r.verdict['signal']} ({r.verdict['rationale']})" for r in subagent_results if r.verdict["signal"] != overall]

    return {
        "overall_signal": overall,
        "confidence": round(sum(r.verdict["confidence"] for r in subagent_results) / len(subagent_results), 2),
        "summary": f"[MOCK] Majority vote across {len(subagent_results)} subagents: {counts}. "
        f"{'Unanimous.' if is_unanimous else ('Tied — defaulted to neutral.' if is_tied else 'Majority, not unanimous.')}",
        "supporting_points": supporting,
        "conflicting_points": conflicting,
        "risks": ["This is a MOCK run — overall_signal is a majority vote over rule-based subagent "
                   "outputs, not real cross-agent reasoning. Run with --live and a real ANTHROPIC_API_KEY "
                   "for an actual synthesized thesis."],
    }
