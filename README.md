# multi-agent-analyst

A supervisor agent that coordinates three narrow subagents — technical,
fundamental, sentiment — each restricted to exactly one data source, and
synthesizes their independent verdicts into one investment thesis with an
explicit audit trail. Built as a real Claude tool-use agent system (not a
single prompt pretending to be three agents), with a fully offline mock
mode for testing the orchestration without an API key or any cost.

**This is not financial advice and isn't trying to be.** The point of this
project is the *architecture* — narrow agents, forced structured output,
an audit trail that shows exactly which data point drove which verdict —
not alpha generation. Treat any output, mock or live, as a demonstration
of a multi-agent pattern, not a signal to trade on.

## Architecture

```
                    ┌─────────────────────┐
                    │   Supervisor Agent    │
                    │  (no data tools —     │
                    │   only synthesizes)   │
                    └──────────┬───────────┘
                               │ reads all three verdicts
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼────────┐ ┌─────▼──────────┐ ┌───▼─────────────┐
   │ Technical Analyst   │ │ Fundamental     │ │ Sentiment        │
   │ Agent                │ │ Analyst Agent   │ │ Analyst Agent    │
   │                      │ │                 │ │                  │
   │ tool: get_technical_ │ │ tool: get_      │ │ tool: get_recent_│
   │       snapshot       │ │ fundamentals_   │ │       headlines  │
   │                      │ │ snapshot        │ │                  │
   └──────────────────────┘ └─────────────────┘ └──────────────────┘
```

**Each subagent gets exactly one data tool — on purpose.** The technical
agent literally cannot call the fundamentals tool, even if it wanted to.
That's what makes the supervisor's synthesis step meaningful: it's
combining three genuinely independent reads on the same ticker, not
re-reading the same blob of context three times with different framing.

**Every agent must finish by calling a `submit_verdict` (or
`submit_thesis`) tool**, with a schema-enforced shape:
`{signal, confidence, rationale, key_evidence}`. This is a deliberate
choice over asking the model to "respond in JSON" as plain text — a tool
call is schema-validated by the API itself, not regex-parsed out of a chat
message that might wrap the JSON in markdown fences or add a sentence
before it. It also gives every agent a clean, unambiguous way to end its
turn.

## Mock mode vs. live mode — read this before looking at any output

| | `--mock` (default) | `--live` |
|---|---|---|
| Cost | Free | Real Anthropic API usage |
| Needs API key | No | Yes (`ANTHROPIC_API_KEY`) |
| What it does | Calls each data tool exactly once, then applies **named, explicit threshold rules** (e.g. "3m return < -5% → bearish") | Real Claude tool-use loop — the model decides what to call and reasons over the result |
| Is it real analysis | **No.** Every mock rationale is prefixed `[MOCK]` and states the literal rule that fired | Yes, to whatever degree an LLM's reasoning over this data counts as "real analysis" — still not investment advice |

Mock mode exists to let you (or me, building this) verify the entire
pipeline — tool dispatch, the audit trail, how the supervisor aggregates
disagreement — without spending money or needing a key. It is structurally
incapable of producing insight; it's a fixed if-statement wearing the same
output schema a real agent would use, so the rest of the system can't tell
the difference and doesn't need to. Don't mistake a mock run for a real
one — the `mode` field on every report and the `[MOCK]` prefix on every
generated string are there specifically so you can't.

## Running it

```bash
pip install -r requirements.txt

# Free, offline, deterministic
python run_analysis.py --ticker AAPL --mock

# Real agent reasoning — costs money, needs a key
export ANTHROPIC_API_KEY=sk-...
python run_analysis.py --ticker AAPL --live
```

Or as an API:
```bash
uvicorn app:app --reload --port 8003
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "mode": "mock"}'
```

### Example output (mock mode, abbreviated)

```
TECHNICAL ANALYST
  signal: bearish
  confidence: 0.5
  rationale: [MOCK] Rule fired: 3m return < -5%. (3m return=-5.5%, RSI14=70.0)

FUNDAMENTAL ANALYST
  signal: neutral
  confidence: 0.5
  rationale: [MOCK] Rule fired: no extreme valuation/growth combination.

SENTIMENT ANALYST
  signal: neutral
  confidence: 0.4
  rationale: [MOCK] Rule fired: tied 4-4 on keyword count.

SUPERVISOR — SYNTHESIZED THESIS
  overall_signal: neutral
  confidence: 0.47
  summary: [MOCK] Majority vote across 3 subagents: {'neutral': 2, 'bearish': 1}.
  conflicting_points: ['technical: bearish (...)']
```

Every block also prints its audit trail — the exact tool call and the data
it returned — directly underneath the verdict.

## Data sources (and their honest limitations)

- **Technical** (`src/market_data.py`): real OHLCV via `yfinance` when
  network access works, with a clearly-labeled synthetic fallback
  otherwise. The indicators themselves (SMA, RSI, realized volatility,
  trailing returns) are all standard and checkable against a normal price
  chart.
- **Fundamentals** (`src/fundamentals.py`): real ratios via
  `yfinance.Ticker.info` when available, synthetic fallback otherwise —
  every synthetic response includes `"source": "synthetic..."` explicitly,
  so nothing downstream can mistake it for a real filing.
- **Sentiment** (`src/headlines.py`): **synthetic by design, not just by
  fallback.** There's no real news API wired in here on purpose — THESIS
  already has a working sentiment pipeline (Cohere with a VADER fallback).
  Building a second, redundant headline-fetching layer would just be
  duplicate infrastructure. Instead:

  ```python
  from src.headlines import set_headline_source

  def my_real_source(ticker: str) -> list[dict]:
      return [{"headline": "...", "date": "2026-06-20"}, ...]

  set_headline_source(my_real_source)
  ```

  Call that once at startup (pointed at THESIS's existing news layer, or
  any real news API) and the sentiment agent's tool calls return real
  headlines — no other code changes.

## Wiring into THESIS

1. Copy `src/` into the THESIS backend (or import this as a sibling
   package) and mount `app.py`'s `/analyze` route under THESIS's existing
   FastAPI app, same pattern as `stock-forecast-bench` and
   `ticker-clustering`.
2. Call `set_headline_source()` once at startup, pointed at THESIS's
   existing Cohere/VADER pipeline's underlying headline fetcher, so the
   sentiment agent reasons over real news instead of synthetic headlines.
3. **Cache live-mode results.** Each `mode="live"` call is several real
   Claude requests (one per subagent, plus the supervisor) — running that
   fresh on every page load is both slow and a real ongoing cost. Reuse
   the exact precompute-and-cache pattern from `stock-forecast-bench`'s
   `app.py`: an APScheduler job runs the full analysis once a day per
   ticker, caches the JSON, and `/analyze` serves the cache unless asked to
   force-refresh.
4. New React tab: four cards (technical / fundamental / sentiment /
   supervisor), each showing its signal, confidence, rationale, and a
   collapsible "audit trail" section listing the tool call and raw data —
   the auditability is the actual point of the UI, not just the backend.

## Project structure

```
multi-agent-analyst/
├── app.py                # FastAPI wrapper
├── run_analysis.py        # CLI entry point
├── src/
│   ├── llm_client.py       # AnthropicLLMClient (real) + MockLLMClient (offline)
│   ├── agents.py            # the 3 subagents + supervisor, system prompts
│   ├── tools.py              # tool schemas + dispatch table
│   ├── market_data.py        # technical indicators (yfinance + fallback)
│   ├── fundamentals.py       # valuation ratios (yfinance + fallback)
│   ├── headlines.py          # headlines (synthetic, pluggable real source)
│   ├── pipeline.py           # orchestrates all four agents into one report
│   └── seed.py                # shared deterministic seeding for synthetic data
└── requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

The suite pins the behaviors that actually caught bugs during development
(see the sections above), not ceremony coverage — every test encodes a
check where the wrong answer was at some point the actual behavior.

## Possible next steps

- Add a fourth subagent for macro/sector context (e.g. how the stock's
  sector ETF has moved) — currently the three subagents are entirely
  ticker-specific with no broader market read.
- Let the supervisor ask a subagent a follow-up question when two verdicts
  conflict sharply, instead of always treating disagreement as a flat list
  — a real analyst would dig into *why* technical and fundamentals
  disagree, not just note that they do.
- Track thesis changes over time per ticker (same idea as
  ticker-clustering's "next steps" — a ticker's thesis flipping from
  bullish to bearish week-over-week is a more interesting signal than any
  single snapshot).
