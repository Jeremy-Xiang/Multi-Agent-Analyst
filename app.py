"""
app.py — FastAPI wrapper for THESIS integration.

    uvicorn app:app --reload --port 8003
    curl -X POST http://localhost:8003/analyze -H "Content-Type: application/json" \
        -d '{"ticker": "AAPL", "mode": "mock"}'

Unlike stock-forecast-bench's app.py, there's no caching layer here by
default: an agent run involves real LLM calls in "live" mode, which cost
money per request, so the right caching strategy depends on how fresh you
need a thesis to be (probably "once a day per ticker" same as the
forecast cache — see that project's app.py for the exact pattern if you
want to reuse it here).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import run_full_analysis

app = FastAPI(title="Multi-Agent Financial Analyst API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    ticker: str
    mode: str = Field("mock", pattern="^(mock|live)$", description="'mock' (offline, free, rule-based) or 'live' (real Claude calls, needs ANTHROPIC_API_KEY)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        return run_full_analysis(req.ticker, mode=req.mode)
    except RuntimeError as exc:
        # Most likely cause: mode="live" with no ANTHROPIC_API_KEY set.
        raise HTTPException(status_code=400, detail=str(exc))
