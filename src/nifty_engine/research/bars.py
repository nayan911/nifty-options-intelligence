"""Session-aware bar builder for research/replay.

Builds 5m/15m bars from validated 1m history the same way the live RealtimeCandleBuilder buckets
them (clock-aligned floors: minute // N * N, so 09:15/09:20/... and 09:15/09:30/...), using only
minutes inside the instrument's trading window (research.sessions.expected_minutes): pre-open
prints (e.g. 09:08-09:14 on 7-9 Sep 2026) never enter any bar or indicator.

Aggregation: open=first, high=max, low=min, close=last, volume=sum, oi=last. Stitched futures
switch contracts only between sessions, and bars never span sessions, so no bar mixes contracts.
"""

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.core.interfaces import Candle
from nifty_engine.research.sessions import expected_minutes

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15}


def in_session_1m(df: pd.DataFrame, cal: ExchangeCalendar, kind: str) -> pd.DataFrame:
    """Keep only 1m rows whose timestamp is an expected session minute for its day."""
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata")
    days = sorted(set(ts.dt.date))
    allowed = set()
    for d in days:
        allowed.update(pd.Timestamp(m) for m in expected_minutes(cal, d, kind))
    mask = ts.isin(allowed)
    out = df.loc[mask].copy()
    out["ts"] = ts[mask]
    return out.sort_values("ts").reset_index(drop=True)


def resample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate in-session 1m rows (columns ts, open, high, low, close, volume, oi) into `tf`."""
    if tf == "1m" or df1m.empty:
        return df1m.copy()
    n = TF_MINUTES[tf]
    ts = pd.to_datetime(df1m["ts"])
    bucket = ts.dt.floor(f"{n}min")
    g = df1m.assign(bucket=bucket).groupby("bucket", sort=True)
    out = pd.DataFrame({
        "ts": g["ts"].first().index,
        "open": g["open"].first().values,
        "high": g["high"].max().values,
        "low": g["low"].min().values,
        "close": g["close"].last().values,
        "volume": g["volume"].sum().values,
        "oi": g["oi"].last().values if "oi" in df1m else None,
    })
    return out.reset_index(drop=True)


def to_candles(df: pd.DataFrame, symbol: str, tf: str) -> List[Candle]:
    n = TF_MINUTES.get(tf)
    out = []
    for r in df.itertuples(index=False):
        ts = r.ts.to_pydatetime() if hasattr(r.ts, "to_pydatetime") else r.ts
        oi = getattr(r, "oi", None)
        oi = None if oi is None or pd.isna(oi) else int(oi)
        close_at = ts + timedelta(minutes=n) if n else ts
        out.append(Candle(symbol=symbol, timeframe=tf, timestamp=ts, open=float(r.open), high=float(r.high),
                          low=float(r.low), close=float(r.close), volume=int(r.volume), oi=oi, is_closed=True,
                          observed_at=close_at))
    return out


def build_intraday_candles(df1m: pd.DataFrame, cal: ExchangeCalendar, kind: str, symbol: str) -> Dict[str, List[Candle]]:
    """{'1m': [...], '5m': [...], '15m': [...]} from raw 1m research rows (session-filtered)."""
    clean = in_session_1m(df1m, cal, kind)
    return {tf: to_candles(resample(clean, tf), symbol, tf) for tf in ("1m", "5m", "15m")}


def daily_candles(df_d: pd.DataFrame, symbol: str) -> List[Candle]:
    out = []
    for r in df_d.itertuples(index=False):
        ts = pd.Timestamp(r.ts).tz_convert("Asia/Kolkata").to_pydatetime()
        oi = getattr(r, "oi", None)
        out.append(Candle(symbol=symbol, timeframe="D", timestamp=ts, open=float(r.open), high=float(r.high),
                          low=float(r.low), close=float(r.close), volume=int(r.volume),
                          oi=None if oi is None or pd.isna(oi) else int(oi), is_closed=True))
    return out


def trading_days(cal: ExchangeCalendar, start: date, end: date) -> List[date]:
    return [d for d in (start + timedelta(days=i) for i in range((end - start).days + 1)) if cal.is_trading_day(d)]
