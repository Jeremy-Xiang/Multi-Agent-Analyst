"""
fundamentals.py — Valuation/quality ratios for the fundamental analyst
agent's tool.

yfinance's `Ticker.info` carries real fundamentals when network access
works (it doesn't in this sandbox — see README). The synthetic fallback
generates *plausible-looking* ratios from a per-ticker deterministic seed,
clearly distinguishable in the output as synthetic via the `"source"`
field, so nothing downstream can mistake fallback data for a real filing.
"""

from __future__ import annotations

import numpy as np

from .seed import stable_seed


def get_fundamentals_snapshot(ticker: str) -> dict:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        if not info or "trailingPE" not in info and "forwardPE" not in info:
            raise RuntimeError("yfinance returned no usable fundamentals")

        return {
            "ticker": ticker,
            "source": "yfinance",
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "profit_margin_pct": _pct(info.get("profitMargins")),
            "revenue_growth_yoy_pct": _pct(info.get("revenueGrowth")),
            "debt_to_equity": info.get("debtToEquity"),
            "dividend_yield_pct": _pct(info.get("dividendYield")),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[fundamentals.py] Live fetch failed for {ticker} ({exc}). Using synthetic fallback.")
        return _synthetic_fundamentals(ticker)


def _pct(x):
    return round(x * 100, 1) if x is not None else None


def _synthetic_fundamentals(ticker: str) -> dict:
    rng = np.random.default_rng(stable_seed(ticker))
    return {
        "ticker": ticker,
        "source": "synthetic (offline fallback — not a real filing)",
        "trailing_pe": round(float(rng.uniform(8, 45)), 1),
        "forward_pe": round(float(rng.uniform(7, 40)), 1),
        "peg_ratio": round(float(rng.uniform(0.5, 3.0)), 2),
        "profit_margin_pct": round(float(rng.uniform(-5, 35)), 1),
        "revenue_growth_yoy_pct": round(float(rng.uniform(-10, 40)), 1),
        "debt_to_equity": round(float(rng.uniform(0.1, 2.5)), 2),
        "dividend_yield_pct": round(float(rng.uniform(0, 4)), 2),
    }
