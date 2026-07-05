"""
market_data.py — Price history + technical indicators for the technical
analyst agent's tool.

Same yfinance-first, synthetic-fallback pattern used throughout this
project's siblings (stock-forecast-bench, ticker-clustering): try live
data, fall back to a clearly-labeled deterministic synthetic series if the
network isn't available, so the pipeline still runs end to end for testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seed import stable_seed


def load_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[market_data.py] Live fetch failed for {ticker} ({exc}). Using synthetic fallback.")
        return _synthetic_price_history(seed=stable_seed(ticker))


def _synthetic_price_history(n_days: int = 280, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    drift = rng.uniform(-0.0004, 0.0010)
    vol = rng.uniform(0.012, 0.032)
    daily_returns = rng.normal(loc=drift, scale=vol, size=n_days)
    close = start_price * np.exp(np.cumsum(daily_returns))

    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, size=n_days - 1))
    intraday_range = np.abs(rng.normal(0.006, 0.004, size=n_days)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range
    volume = rng.integers(500_000, 50_000_000, size=n_days)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
    )


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean().iloc[-1]
    avg_loss = loss.rolling(window).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def get_technical_snapshot(ticker: str) -> dict:
    """
    The technical analyst agent's only tool. Everything in here is a
    standard, checkable indicator — nothing exotic — so a human can verify
    any of these numbers against a normal price chart.
    """
    df = load_price_history(ticker, period="1y")
    close = df["Close"]

    latest_close = float(close.iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    rsi_14 = _rsi(close, 14)

    daily_returns = close.pct_change().dropna()
    volatility_21d_annualized = float(daily_returns.tail(21).std() * np.sqrt(252))

    def trailing_return(n_days: int):
        if len(close) <= n_days:
            return None
        return float(close.iloc[-1] / close.iloc[-n_days] - 1)

    high_52w = float(close.tail(252).max()) if len(close) >= 1 else None
    low_52w = float(close.tail(252).min()) if len(close) >= 1 else None

    return {
        "ticker": ticker,
        "latest_close": round(latest_close, 2),
        "sma_20": round(sma_20, 2) if sma_20 else None,
        "sma_50": round(sma_50, 2) if sma_50 else None,
        "rsi_14": round(rsi_14, 1),
        "volatility_21d_annualized_pct": round(volatility_21d_annualized * 100, 1),
        "trailing_return_1m_pct": round(trailing_return(21) * 100, 1) if trailing_return(21) is not None else None,
        "trailing_return_3m_pct": round(trailing_return(63) * 100, 1) if trailing_return(63) is not None else None,
        "trailing_return_6m_pct": round(trailing_return(126) * 100, 1) if trailing_return(126) is not None else None,
        "pct_from_52w_high": round((latest_close / high_52w - 1) * 100, 1) if high_52w else None,
        "pct_from_52w_low": round((latest_close / low_52w - 1) * 100, 1) if low_52w else None,
    }
