"""Research indicator library (Phase 2.4): the 10 agreed indicators, OBSERVE-ONLY.

Nothing here feeds S000's decisions. The same functions serve live recording and historical
research: callers pass one session's 1-minute bars (plus a little prior context) and get a
per-minute feature table.

Causality contract (tested by truncation: features at minute m computed from the full day equal
features computed from only the bars that had closed by then):
- Row `m` describes the 1m bar that STARTS at m; its values become available at m + 1 minute
  (`available_at`). Every value in a row uses only bars starting at or before m.
- Prior-session context (RVOL profile, daily ATR, previous VIX close) comes strictly from earlier
  sessions.
- When a value cannot be known yet or the input is missing it is NaN/UNKNOWN, with a reason code
  in `unavailable` (e.g. before_or_complete, insufficient_rvol_history, next_month_missing).

Indicators (columns):
 1 vwap, vwap_sigma, vwap_z      futures session VWAP (engine's compute_vwap: typical price,
                                 volume-weighted) and the volume-weighted price dispersion; z =
                                 (futures close - vwap) / sigma
 2 rvol_tod_1m, rvol_tod_5m      futures volume (near + next month, so rollover week is not
                                 distorted) of the last 1 / 5 completed minutes vs the mean of the
                                 same minutes over the previous `RVOL_SESSIONS` regular sessions
 3 or_width_atr                  spot opening range (first 15 minutes) / spot daily ATR(14) of
                                 previous sessions; UNKNOWN until the range is complete
 4 fut_oi_chg_open_pct, fut_oi_chg_15m_pct
                                 near-month OI change since the session's first bar / last 15 min.
                                 Intraday only: a stitched series changes contract only between
                                 sessions, so this never compares two contracts
 5 next_oi_chg_open_pct, comb_oi_chg_open_pct, rollover_window
                                 next-month OI change and combined near+next OI change since the
                                 open; rollover_window = within ROLLOVER_DAYS trading days of the
                                 near expiry (when the combined basket's meaning shifts)
 6 basis, basis_pct, basis_chg_15m
                                 futures close - spot close at the same minute (no carry-over)
 7 vix, vix_chg_prev_close_pct, vix_chg_15m
 8 days_to_expiry, expiry_day    trading days to the near contract's (holiday-adjusted) expiry
 9 hw_up_count, hw_agree_15m     heavyweights (5 stocks): how many are above their session open;
                                 how many moved in the same direction as NIFTY spot over 15 min.
                                 A five-stock confirmation, not market breadth
10 day_type, day_type_stage      TRENDING_UP / TRENDING_DOWN / RANGE / MIXED / UNKNOWN, built only
                                 from information so far; stage provisional (OR complete -> +30m),
                                 early (-> +90m), established after that
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from nifty_engine.analysis.technical.indicators import compute_atr, compute_vwap

RVOL_SESSIONS = 10
RVOL_MIN_SESSIONS = 5
OR_MINUTES = 15
ROLLOVER_DAYS = 3
DAILY_ATR_PERIOD = 14
LOOKBACK = 15            # minutes for *_15m changes
DAY_TYPE_WINDOW = 30     # minutes for VWAP side/slope in the day-type classifier
FEATURE_VERSION = 1

FEATURE_COLUMNS = [
    "vwap", "vwap_sigma", "vwap_z",
    "rvol_tod_1m", "rvol_tod_5m",
    "or_width_atr",
    "fut_oi_chg_open_pct", "fut_oi_chg_15m_pct",
    "next_oi_chg_open_pct", "comb_oi_chg_open_pct", "rollover_window",
    "basis", "basis_pct", "basis_chg_15m",
    "vix", "vix_chg_prev_close_pct", "vix_chg_15m",
    "days_to_expiry", "expiry_day",
    "hw_up_count", "hw_agree_15m",
    "day_type", "day_type_stage",
]


@dataclass
class DayInputs:
    """One session's inputs. Bar frames: columns ts (bar start, tz-aware), open, high, low, close,
    volume, [oi]; in-session minutes only; ascending."""
    day: date
    spot: pd.DataFrame
    fut: pd.DataFrame
    fut_next: Optional[pd.DataFrame] = None
    vix: Optional[pd.DataFrame] = None
    heavyweights: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # prior context (strictly earlier sessions)
    prior_volume_profiles: List[pd.Series] = field(default_factory=list)  # index: minute-of-day (int)
    prior_spot_daily: Optional[pd.DataFrame] = None   # daily bars before `day` (open/high/low/close)
    prev_vix_close: Optional[float] = None
    days_to_expiry: Optional[int] = None
    special_session: bool = False


def minute_of_day(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    return t.dt.hour * 60 + t.dt.minute


def volume_profile(fut: pd.DataFrame, fut_next: Optional[pd.DataFrame] = None) -> pd.Series:
    """Per-minute-of-day futures volume (near + next) of one session, for later RVOL baselines."""
    v = fut.set_index(minute_of_day(fut["ts"]))["volume"].astype(float)
    if fut_next is not None and not fut_next.empty:
        nv = fut_next.set_index(minute_of_day(fut_next["ts"]))["volume"].astype(float)
        v = v.add(nv, fill_value=0.0)
    return v.groupby(level=0).sum()


def _on(index: pd.DatetimeIndex, df: Optional[pd.DataFrame], col: str) -> pd.Series:
    """Column of `df` aligned to `index` WITHOUT carry-over (missing minute -> NaN)."""
    if df is None or df.empty:
        return pd.Series(np.nan, index=index)
    return df.set_index(pd.to_datetime(df["ts"]))[col].astype(float).reindex(index)


def _ffill(index: pd.DatetimeIndex, df: Optional[pd.DataFrame], col: str) -> pd.Series:
    """Latest known value of `df[col]` at each minute (causal carry-forward)."""
    return _on(index, df, col).ffill()


def compute_day_features(inp: DayInputs) -> pd.DataFrame:
    """Per-minute feature table for one session (see module docstring for the causality contract)."""
    idx = pd.DatetimeIndex(sorted(set(pd.to_datetime(inp.spot["ts"])) | set(pd.to_datetime(inp.fut["ts"]))))
    out = pd.DataFrame(index=idx)
    out.index.name = "ts"
    reasons: Dict[str, str] = {}

    # ---- 1. VWAP bands (futures) - the engine's own compute_vwap
    fut = inp.fut.copy()
    fut_close = _ffill(idx, fut, "close")
    if not fut.empty and fut["volume"].sum() > 0:
        f = fut.reset_index(drop=True)
        vwap, up1, _ = compute_vwap(f)
        s = pd.Series(vwap.values, index=pd.to_datetime(f["ts"])).reindex(idx).ffill()
        sig = pd.Series((up1 - vwap).values, index=pd.to_datetime(f["ts"])).reindex(idx).ffill()
        out["vwap"], out["vwap_sigma"] = s, sig
        out["vwap_z"] = (fut_close - s) / sig.where(sig > 0)
    else:
        out["vwap"] = out["vwap_sigma"] = out["vwap_z"] = np.nan
        reasons["vwap"] = reasons["vwap_sigma"] = reasons["vwap_z"] = "no_futures_volume"

    # ---- 2. time-of-day RVOL (near + next volume vs the same minutes in previous sessions)
    vol_now = _on(idx, fut, "volume")  # a missing near-month minute stays unknown (not zero volume)
    if inp.fut_next is not None and not inp.fut_next.empty:
        vol_now = vol_now + _on(idx, inp.fut_next, "volume").fillna(0.0)
    mod = pd.Series(idx.hour * 60 + idx.minute, index=idx)
    profiles = inp.prior_volume_profiles[-RVOL_SESSIONS:]
    if len(profiles) >= RVOL_MIN_SESSIONS:
        prof = pd.concat(profiles, axis=1)
        base1 = mod.map(prof.mean(axis=1))
        roll5 = prof.sort_index().rolling(5, min_periods=5).sum().mean(axis=1)
        base5 = mod.map(roll5)
        out["rvol_tod_1m"] = vol_now / base1.where(base1 > 0)
        out["rvol_tod_5m"] = vol_now.rolling(5, min_periods=5).sum() / base5.where(base5 > 0)
    else:
        out["rvol_tod_1m"] = out["rvol_tod_5m"] = np.nan
        reasons["rvol_tod_1m"] = reasons["rvol_tod_5m"] = "insufficient_rvol_history"

    # ---- 3. opening range width / daily ATR (spot)
    spot_close = _ffill(idx, inp.spot, "close")
    sp = inp.spot.set_index(pd.to_datetime(inp.spot["ts"])) if not inp.spot.empty else None
    or_hi = or_lo = None
    atr_d = None
    if inp.prior_spot_daily is not None and len(inp.prior_spot_daily) > DAILY_ATR_PERIOD:
        d = inp.prior_spot_daily
        atr_d = float(compute_atr(d["high"].astype(float), d["low"].astype(float), d["close"].astype(float),
                                  DAILY_ATR_PERIOD).iloc[-1])
    out["or_width_atr"] = np.nan
    if sp is None or inp.special_session:
        reasons["or_width_atr"] = "special_session" if inp.special_session else "no_spot"
    elif atr_d is None or not atr_d > 0:
        reasons["or_width_atr"] = "insufficient_daily_history"
    else:
        first = sp.index.min()
        or_end = first + pd.Timedelta(minutes=OR_MINUTES)
        window = sp[(sp.index >= first) & (sp.index < or_end)]
        if len(window) >= OR_MINUTES - 2:
            or_hi, or_lo = float(window["high"].max()), float(window["low"].min())
            known = idx >= or_end - pd.Timedelta(minutes=1)  # row of the last OR bar: complete at its close
            out.loc[known, "or_width_atr"] = (or_hi - or_lo) / atr_d
        reasons["or_width_atr"] = "before_or_complete"

    # ---- 4. near-month OI change (intraday, same contract)
    oi = _ffill(idx, fut, "oi") if "oi" in fut else pd.Series(np.nan, index=idx)
    oi_open = oi.dropna().iloc[0] if oi.notna().any() else np.nan
    out["fut_oi_chg_open_pct"] = 100 * (oi - oi_open) / oi_open if oi_open and oi_open > 0 else np.nan
    out["fut_oi_chg_15m_pct"] = 100 * (oi / oi.shift(LOOKBACK) - 1)
    if not (oi_open and oi_open > 0):
        reasons["fut_oi_chg_open_pct"] = reasons["fut_oi_chg_15m_pct"] = "no_futures_oi"

    # ---- 5. next-month and combined OI
    if inp.fut_next is not None and not inp.fut_next.empty and "oi" in inp.fut_next:
        noi = _ffill(idx, inp.fut_next, "oi")
        noi_open = noi.dropna().iloc[0] if noi.notna().any() else np.nan
        out["next_oi_chg_open_pct"] = 100 * (noi - noi_open) / noi_open if noi_open and noi_open > 0 else np.nan
        comb = oi + noi
        comb_open = comb.dropna().iloc[0] if comb.notna().any() else np.nan
        out["comb_oi_chg_open_pct"] = 100 * (comb - comb_open) / comb_open if comb_open and comb_open > 0 else np.nan
    else:
        out["next_oi_chg_open_pct"] = out["comb_oi_chg_open_pct"] = np.nan
        reasons["next_oi_chg_open_pct"] = reasons["comb_oi_chg_open_pct"] = "next_month_missing"
    dte = inp.days_to_expiry
    out["rollover_window"] = (dte is not None and dte <= ROLLOVER_DAYS)

    # ---- 6. basis at matched minutes (no carry-over between the two series)
    fc, sc = _on(idx, fut, "close"), _on(idx, inp.spot, "close")
    basis = fc - sc
    out["basis"] = basis
    out["basis_pct"] = 100 * basis / sc
    out["basis_chg_15m"] = basis - basis.shift(LOOKBACK)

    # ---- 7. India VIX
    vix = _ffill(idx, inp.vix, "close")
    out["vix"] = vix
    out["vix_chg_prev_close_pct"] = (100 * (vix / inp.prev_vix_close - 1)) if inp.prev_vix_close else np.nan
    out["vix_chg_15m"] = vix - vix.shift(LOOKBACK)
    if inp.vix is None or inp.vix.empty:
        reasons["vix"] = reasons["vix_chg_15m"] = reasons["vix_chg_prev_close_pct"] = "no_vix"
    elif not inp.prev_vix_close:
        reasons["vix_chg_prev_close_pct"] = "no_prev_vix_close"

    # ---- 8. expiry
    out["days_to_expiry"] = dte if dte is not None else np.nan
    out["expiry_day"] = dte == 0
    if dte is None:
        reasons["days_to_expiry"] = "unknown_expiry"

    # ---- 9. heavyweights
    if inp.heavyweights:
        nifty_ret = spot_close - spot_close.shift(LOOKBACK)
        up = pd.Series(0, index=idx, dtype=float)
        agree = pd.Series(0, index=idx, dtype=float)
        for _, hw in sorted(inp.heavyweights.items()):
            c = _ffill(idx, hw, "close")
            o = hw["open"].iloc[0] if not hw.empty else np.nan
            up += (c > o).astype(float)
            r = c - c.shift(LOOKBACK)
            agree += ((np.sign(r) == np.sign(nifty_ret)) & (np.sign(r) != 0)).astype(float)
        out["hw_up_count"] = up
        out["hw_agree_15m"] = agree.where(nifty_ret.notna())
    else:
        out["hw_up_count"] = out["hw_agree_15m"] = np.nan
        reasons["hw_up_count"] = reasons["hw_agree_15m"] = "no_heavyweights"

    # ---- 10. day type (causal, staged)
    out["day_type"], out["day_type_stage"] = _day_type(idx, spot_close, fut_close, out["vwap"], or_hi, or_lo, atr_d)

    out["unavailable"] = [
        ";".join(f"{c}:{reasons.get(c, 'warming_up')}" for c in FEATURE_COLUMNS
                 if c not in ("day_type", "day_type_stage") and pd.isna(row[c]))
        for _, row in out[FEATURE_COLUMNS].iterrows()
    ]
    out["available_at"] = out.index + pd.Timedelta(minutes=1)
    out["feature_version"] = FEATURE_VERSION
    return out


def _day_type(idx, spot_close, fut_close, vwap, or_hi, or_lo, atr_d):
    """Rules (fixed a priori; experimental feature):
    UNKNOWN until the opening range is complete (or without a range / ATR).
    side  = share of the last DAY_TYPE_WINDOW futures closes on the current side of VWAP.
    slope = VWAP change over the window / daily ATR.
    TRENDING_UP   spot above OR high, side >= 0.8, slope > +0.03
    TRENDING_DOWN spot below OR low,  side >= 0.8, slope < -0.03
    RANGE         spot inside the OR and side < 0.6
    MIXED         otherwise
    """
    n = len(idx)
    types = ["UNKNOWN"] * n
    stages = ["none"] * n
    if or_hi is None or atr_d is None or not atr_d > 0 or n == 0:
        return types, stages
    first = idx.min()
    complete = first + pd.Timedelta(minutes=OR_MINUTES - 1)
    above = (fut_close > vwap).astype(float).where(vwap.notna())
    for i, ts in enumerate(idx):
        if ts < complete or pd.isna(spot_close.iloc[i]) or pd.isna(vwap.iloc[i]):
            continue
        lo = max(0, i - DAY_TYPE_WINDOW + 1)
        win = above.iloc[lo:i + 1].dropna()
        if len(win) < 10:
            continue
        cur = above.iloc[i]
        side = float((win == cur).mean())
        slope = (vwap.iloc[i] - vwap.iloc[lo]) / atr_d
        p = spot_close.iloc[i]
        if p > or_hi and side >= 0.8 and slope > 0.03:
            t = "TRENDING_UP"
        elif p < or_lo and side >= 0.8 and slope < -0.03:
            t = "TRENDING_DOWN"
        elif or_lo <= p <= or_hi and side < 0.6:
            t = "RANGE"
        else:
            t = "MIXED"
        elapsed = (ts - complete).total_seconds() / 60
        types[i] = t
        stages[i] = "provisional" if elapsed < 30 else ("early" if elapsed < 90 else "established")
    return types, stages


def features_at(table: pd.DataFrame, t: datetime) -> Optional[pd.Series]:
    """The latest feature row available at time t (row available_at <= t), or None."""
    avail = table[table["available_at"] <= pd.Timestamp(t)]
    return None if avail.empty else avail.iloc[-1]


def trading_days_between(cal, start: date, end: date) -> int:
    """Trading sessions strictly after `start` up to and including `end` (0 if same day)."""
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if cal.is_trading_day(d):
            n += 1
    return n
