"""Phase 3B: build the research feature dataset from validated history.

For every trading day, the 10 research indicators (features.library) are computed per minute from
that day's research bars plus strictly-earlier context. They are then attached, using only rows
already available at the observation time, at three levels:
  bars           every closed 5-minute bar of the session (market conditions, unconditional)
  setups         every raw S000 setup (available at its bar close)
  opportunities  every S000 opportunity (at its decision time), joined with its labelled outcomes
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.features.library import (
    FEATURE_COLUMNS,
    RVOL_SESSIONS,
    DayInputs,
    compute_day_features,
    features_at,
    trading_days_between,
    volume_profile,
)
from nifty_engine.research.bars import build_intraday_candles, in_session_1m, trading_days
from nifty_engine.research.contracts import contract_expiry_for
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.sessions import CASH, FNO

HEAVYWEIGHTS = ("HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "BHARTIARTL")
TF_MIN = {"1m": 1, "5m": 5, "15m": 15}


class FeatureDatasetBuilder:
    def __init__(self, store: ResearchHistoryStore, calendar: Optional[ExchangeCalendar] = None) -> None:
        self.store = store
        self.cal = calendar or ExchangeCalendar()
        self._frames: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ loading
    def _load(self, key: str, instrument: str, source: str, symbol: Optional[str], kind: str) -> pd.DataFrame:
        q = ("SELECT ts, open, high, low, close, volume, oi FROM hist_bars WHERE instrument = ? AND source = ? "
             "AND resolution = '1'" + (" AND symbol = ?" if symbol else "") + " ORDER BY ts")
        df = self.store.conn.execute(q, [instrument, source] + ([symbol] if symbol else [])).df()
        df = in_session_1m(df, self.cal, kind)
        df["day"] = pd.to_datetime(df["ts"]).dt.date
        self._frames[key] = df
        return df

    def load(self) -> None:
        self._load("spot", "NIFTY_SPOT", "SPOT", None, CASH)
        self._load("fut", "NIFTY_FUT", "CONT", "CONT:NIFTY:NEAR", FNO)
        self._load("next", "NIFTY_FUT", "CONT_NEXT", "CONT:NIFTY:NEXT", FNO)
        self._load("vix", "INDIAVIX", "SPOT", None, CASH)
        for hw in HEAVYWEIGHTS:
            self._load(hw, hw, "SPOT", None, CASH)
        self.spot_daily = self.store.conn.execute(
            "SELECT ts, open, high, low, close FROM hist_bars WHERE instrument = 'NIFTY_SPOT' AND source = 'SPOT' "
            "AND resolution = 'D' ORDER BY ts").df()
        self.spot_daily["day"] = pd.to_datetime(self.spot_daily["ts"]).dt.tz_convert("Asia/Kolkata").dt.date
        vixd = self.store.conn.execute(
            "SELECT ts, close FROM hist_bars WHERE instrument = 'INDIAVIX' AND source = 'SPOT' AND resolution = 'D' "
            "ORDER BY ts").df()
        vixd["day"] = pd.to_datetime(vixd["ts"]).dt.tz_convert("Asia/Kolkata").dt.date
        self.vix_close = dict(zip(vixd["day"], vixd["close"]))
        self._by_day = {k: dict(tuple(v.groupby("day"))) for k, v in self._frames.items()}

    def _day(self, key: str, d: date) -> Optional[pd.DataFrame]:
        g = self._by_day[key].get(d)
        return None if g is None or g.empty else g.drop(columns="day").reset_index(drop=True)

    # ------------------------------------------------------------------ per-day features
    def day_inputs(self, d: date, all_days: List[date]) -> Optional[DayInputs]:
        spot, fut = self._day("spot", d), self._day("fut", d)
        if spot is None or fut is None:
            return None
        prior = [x for x in all_days if x < d]
        regular = [x for x in prior if x not in self.cal.special_sessions]
        profiles = []
        for x in regular[-RVOL_SESSIONS:]:
            f = self._day("fut", x)
            if f is not None:
                profiles.append(volume_profile(f, self._day("next", x)))
        prev_days = [x for x in prior if x in self.vix_close]
        expiry = contract_expiry_for(self.cal, d, "CONT")
        return DayInputs(
            day=d, spot=spot, fut=fut, fut_next=self._day("next", d), vix=self._day("vix", d),
            heavyweights={h: df for h in HEAVYWEIGHTS if (df := self._day(h, d)) is not None},
            prior_volume_profiles=profiles,
            prior_spot_daily=self.spot_daily[self.spot_daily["day"] < d].reset_index(drop=True),
            prev_vix_close=float(self.vix_close[prev_days[-1]]) if prev_days else None,
            days_to_expiry=trading_days_between(self.cal, d, expiry) if expiry else None,
            special_session=d in self.cal.special_sessions,
        )

    def spot_5m_closes(self) -> List:
        """observed_at of every 5m spot bar the live-equivalent builder emits."""
        spot = self._frames["spot"].drop(columns="day")
        return [c.observed_at for c in build_intraday_candles(spot, self.cal, CASH, "NSE:NIFTY50-INDEX")["5m"]]

    def minute_tables(self, start: date, end: date) -> Dict[date, pd.DataFrame]:
        days = sorted(self._by_day["spot"])
        out = {}
        for d in trading_days(self.cal, start, end):
            inp = self.day_inputs(d, days)
            if inp is not None:
                out[d] = compute_day_features(inp)
        return out


# ------------------------------------------------------------------ attachment
def _row(tables: Dict[date, pd.DataFrame], t) -> Optional[pd.Series]:
    ts = pd.Timestamp(t)
    tbl = tables.get(ts.date())
    return None if tbl is None else features_at(tbl, ts)


def _attach(records: List[dict], tables: Dict[date, pd.DataFrame], time_key: str) -> pd.DataFrame:
    rows = []
    for rec in records:
        r = _row(tables, rec[time_key])
        feats = {c: (r[c] if r is not None else None) for c in FEATURE_COLUMNS}
        feats["feature_row_ts"] = r.name if r is not None else None
        feats["unavailable"] = r["unavailable"] if r is not None else "all:no_feature_row"
        rows.append({**rec, **feats})
    return pd.DataFrame(rows)


def bar_level(tables: Dict[date, pd.DataFrame], spot_5m_closes: List) -> pd.DataFrame:
    """Every 5m NIFTY spot bar the live builder actually emits (research.bars, live drop rules
    included): features as of its close."""
    recs = [{"day": pd.Timestamp(t).date(), "observed_at": pd.Timestamp(t)} for t in spot_5m_closes
            if pd.Timestamp(t).date() in tables]
    return _attach(recs, tables, "observed_at")


def setup_level(setups, tables) -> pd.DataFrame:
    recs = [{"day": s.day, "setup_at": s.at, "observed_at": pd.Timestamp(s.at) + timedelta(minutes=TF_MIN[s.timeframe]),
             "timeframe": s.timeframe, "direction": s.direction, "level": s.detail.get("level_id"),
             "trend": s.detail.get("trend")} for s in setups]
    return _attach(recs, tables, "observed_at")


def opportunity_level(baseline_report: dict, tables) -> pd.DataFrame:
    recs = []
    for o in baseline_report["opportunities"]:
        rec = {k: o[k] for k in ("day", "decision_at", "direction", "tier", "n_candidates", "level", "confluence",
                                 "s000_stop")}
        for key, res in o["outcomes"].items():
            rec[f"out_{key}"] = res.get("outcome")
            if "mfe" in res:
                rec[f"mfe_{key}"], rec[f"mae_{key}"] = res["mfe"], res["mae"]
            rec[f"stop_{key}"] = res.get("stop_points")
            rec[f"close_{key}"] = res.get("close_points")
        recs.append(rec)
    return _attach(recs, tables, "decision_at")


def check_baseline_compatible(baseline: dict, dataset_fp: str, strategy) -> None:
    """Outcomes may only be joined to features computed from the SAME data and strategy version."""
    meta = baseline["meta"]
    if meta.get("dataset_fingerprint") != dataset_fp:
        raise ValueError(f"baseline was built on dataset {meta.get('dataset_fingerprint')}, current data is "
                         f"{dataset_fp}: regenerate the baseline (scripts/run_baseline.py) first")
    s = meta.get("strategy", {})
    if (s.get("id"), s.get("version"), s.get("fingerprint")) != (strategy.id, strategy.version,
                                                                 strategy.fingerprint()):
        raise ValueError(f"baseline strategy {s.get('id')} v{s.get('version')} {s.get('fingerprint')} does not match "
                         f"the registry ({strategy.id} v{strategy.version} {strategy.fingerprint()}): "
                         "regenerate the baseline first")
