"""
headlines.py — Recent headlines for the sentiment analyst agent's tool.

There's no real news API wired in here on purpose: THESIS already has a
working sentiment pipeline (Cohere with a VADER fallback). Rather than
build a second, redundant headline-fetching layer, this module is designed
to be swapped out — `set_headline_source()` lets you point it at THESIS's
existing data layer (or any news API) without touching the agent code at
all, since the agent only ever calls `get_recent_headlines(ticker)`.

Until you wire that in, it returns clearly-labeled synthetic headlines —
deterministic per ticker, with a sentiment skew baked into the seed so
different tickers produce noticeably different (but fake) sentiment
profiles for testing the pipeline end to end.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .seed import stable_seed

_POSITIVE_TEMPLATES = [
    "{company} beats quarterly earnings estimates, shares climb",
    "Analysts raise price target on {company} after strong guidance",
    "{company} announces expanded buyback program",
    "{company} unveils new product line, investors react positively",
    "{company} posts record revenue for the quarter",
]
_NEUTRAL_TEMPLATES = [
    "{company} to report earnings next week",
    "{company} holds annual shareholder meeting",
    "What analysts are watching for in {company}'s next report",
    "{company} announces executive leadership change",
    "{company} reaffirms full-year guidance",
]
_NEGATIVE_TEMPLATES = [
    "{company} shares slide after disappointing guidance",
    "{company} faces regulatory scrutiny over recent practices",
    "Analysts cut price target on {company} citing margin pressure",
    "{company} announces layoffs amid restructuring",
    "{company} misses revenue expectations for the quarter",
]

_HeadlineSource = Optional[Callable[[str], list[dict]]]
_custom_source: _HeadlineSource = None


def set_headline_source(fn: Callable[[str], list[dict]]) -> None:
    """
    Point this module at a real headline source. `fn(ticker)` should return
    a list of {"headline": str, "date": "YYYY-MM-DD"} dicts, most recent
    first. Once set, get_recent_headlines() uses it instead of the
    synthetic generator — no other code needs to change.
    """
    global _custom_source
    _custom_source = fn


def get_recent_headlines(ticker: str, n: int = 8) -> dict:
    if _custom_source is not None:
        headlines = _custom_source(ticker)
        return {"ticker": ticker, "source": "custom", "headlines": headlines[:n]}

    return {
        "ticker": ticker,
        "source": "synthetic (offline fallback — not real news)",
        "headlines": _synthetic_headlines(ticker, n),
    }


def _synthetic_headlines(ticker: str, n: int) -> list[dict]:
    rng = np.random.default_rng(stable_seed(ticker))
    # Each ticker gets a fixed sentiment skew so its synthetic headline mix
    # is internally consistent rather than uniformly random noise.
    skew = rng.uniform(-1, 1)  # -1 = bearish-skewed, +1 = bullish-skewed
    weights = np.array(
        [
            max(0.05, 0.33 + skew * 0.25),  # positive
            0.34,  # neutral
            max(0.05, 0.33 - skew * 0.25),  # negative
        ]
    )
    weights /= weights.sum()

    pools = [_POSITIVE_TEMPLATES, _NEUTRAL_TEMPLATES, _NEGATIVE_TEMPLATES]
    headlines = []
    dates = pd_bdate_range_desc(n)
    for i in range(n):
        pool_idx = rng.choice(3, p=weights)
        template = rng.choice(pools[pool_idx])
        headlines.append({"headline": template.format(company=ticker), "date": dates[i]})
    return headlines


def pd_bdate_range_desc(n: int) -> list[str]:
    import pandas as pd

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return [d.strftime("%Y-%m-%d") for d in dates[::-1]]
