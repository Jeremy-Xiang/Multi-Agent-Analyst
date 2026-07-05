# multi-agent-analyst

Three subagents — technical, fundamental, sentiment — each restricted to exactly one data tool, synthesized by a supervisor into a single investment thesis. The restriction is structural: the technical agent's tool schema only includes `get_technical_snapshot`. It can't call `get_fundamentals_snapshot` even if it tries. That's what makes the supervisor's synthesis step meaningful — it's combining three genuinely independent reads, not the same data repackaged three times.

## Architecture

```
Supervisor
├── Technical agent   → get_technical_snapshot (RSI, SMA, momentum, vol)
├── Fundamental agent → get_fundamentals_snapshot (P/E, margins, growth, leverage)
└── Sentiment agent   → get_recent_headlines
```

Each subagent calls its one data tool, then calls `submit_verdict` with a schema-enforced verdict: `{signal, confidence, rationale, key_evidence}`. The supervisor sees all three verdicts, has no data tools of its own, and calls `submit_thesis`. It can't go re-derive a technical read — it has to engage with what the subagents said.

Structured output via tool calls, not "respond in JSON" instructions. A tool call is schema-validated by the API. A JSON instruction in a system prompt is a regex job waiting to fail.

## Mock mode vs. live mode

Mock mode (`--mock`, default) calls each data tool once and applies named threshold rules — "3m return < -5% → bearish." Every mock rationale is prefixed `[MOCK]` and states the exact rule that fired. It's designed to test the pipeline's plumbing — tool dispatch, the audit trail, supervisor aggregation — without API calls or cost. It's not analysis.

Live mode (`--live`) runs real Claude tool-use loops. The model decides what to call, reasons over the results, and produces an actual synthesized thesis. Requires `ANTHROPIC_API_KEY`.

Both produce the same output schema. The rest of the code doesn't need to know which one ran.

## Running it

```bash
pip install -r requirements.txt

python run_analysis.py --ticker AAPL --mock   # free, offline, deterministic
export ANTHROPIC_API_KEY=sk-...
python run_analysis.py --ticker AAPL --live   # real Claude calls
```

Or as an API:

```bash
uvicorn app:app --port 8003
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "mode": "mock"}'
```

`mode="live"` without a key returns a clean `400` with the error, not a crash.

## Example output (mock, abbreviated)

```
TECHNICAL ANALYST
  signal: bearish
  rationale: [MOCK] Rule fired: 3m return < -5%. (3m return=-5.5%, RSI14=70.0)

FUNDAMENTAL ANALYST
  signal: neutral
  rationale: [MOCK] Rule fired: no extreme valuation/growth combination.

SUPERVISOR — SYNTHESIZED THESIS
  overall_signal: neutral
  summary: [MOCK] Majority vote: {'neutral': 2, 'bearish': 1}.
  conflicting_points: ['technical: bearish (...)']
```

Every block prints the full audit trail below the verdict — which tool was called, with what input, returning what data.

## Data sources

Technical and fundamental data use yfinance with a synthetic fallback for offline use. Headlines are synthetic by design — `set_headline_source()` in `src/headlines.py` lets you swap in any real news feed (a function that takes a ticker and returns `[{headline, date}]`) without touching the agent code. THESIS's existing Cohere/VADER pipeline can plug in there directly.

## Running the tests

```bash
pytest tests/ -v
```

Eight tests. The most useful ones: each subagent's trace must contain exactly its own tool and nothing else, the supervisor's trace must be empty, and the 3-way-tie case (one bullish, one neutral, one bearish) must produce neutral.

## Structure

```
multi-agent-analyst/
├── run_analysis.py  # CLI
├── app.py           # FastAPI wrapper
├── src/
│   ├── agents.py    # the 3 subagents + supervisor, system prompts
│   ├── tools.py     # tool schemas + dispatch table
│   ├── llm_client.py # AnthropicLLMClient (real) + MockLLMClient (offline)
│   ├── pipeline.py  # orchestrates all four into one report
│   ├── market_data.py
│   ├── fundamentals.py
│   ├── headlines.py # pluggable headline source
│   └── seed.py
└── tests/test_core.py
```
