"""Contract-switch detection for FYERS stitched futures series, by majority vote.

A stitched series silently changes the underlying contract. Each overnight transition
(previous session's last bar -> this session's first bar) is tested with three independent
signals:

  calendar  this session is the first after a monthly expiry
  oi        OI ratio outside [oi_lo, 1/oi_lo] (positions don't double or halve overnight)
  basis     median futures-minus-spot basis (last 15 min vs next session's first 15 min) jumps by
            more than `basis_jump_pct` (0.4%) of spot: a different expiry carries a different
            cost-of-carry (observed switches: 189-330 pts; normal overnight drift is far smaller)

A boundary is ACCEPTED with >= 2 votes; a single vote is recorded as DISPUTED for review.
Evidence from the 2026 data: the OI rule missed the NIFTY near-month Feb switch (OI only 1.6x,
basis -13 -> +158) and the calendar missed a BANKNIFTY next-month switch FYERS made on 25-Sep
(OI 2.1x, basis 433 -> 156). No single rule is reliable; two agreeing rules were in every case.
"""

from datetime import date
from typing import Dict, List

import pandas as pd

from nifty_engine.core.calendar import ExchangeCalendar
from nifty_engine.research.history_store import ResearchHistoryStore
from nifty_engine.research.sessions import monthly_expiries

SPOT_FOR = {"NIFTY_FUT": "NIFTY_SPOT", "BANKNIFTY_FUT": "BANKNIFTY_SPOT"}


def _day_edges(store: ResearchHistoryStore, instrument: str, source: str, spot_instrument: str,
               edge_bars: int = 15) -> pd.DataFrame:
    """Per session: first/last bar OI and basis (futures close - spot close at the same minute)."""
    df = store.conn.execute(
        """SELECT f.ts, f.close AS fut, f.oi, s.close AS spot
           FROM hist_bars f JOIN hist_bars s
             ON s.ts = f.ts AND s.instrument = ? AND s.resolution = '1'
           WHERE f.instrument = ? AND f.source = ? AND f.resolution = '1'
           ORDER BY f.ts""",
        [spot_instrument, instrument, source],
    ).df()
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata").dt.date
    df["basis"] = df["fut"] - df["spot"]
    g = df.groupby("day")
    # Robust basis: median over the first/last `edge_bars` minutes. A single opening bar is
    # noisy (the index opens from the pre-open auction while futures trade continuously).
    return pd.DataFrame({
        "first_oi": g["oi"].first(), "last_oi": g["oi"].last(),
        "first_basis": g["basis"].apply(lambda x: x.head(edge_bars).median()),
        "last_basis": g["basis"].apply(lambda x: x.tail(edge_bars).median()),
        "spot": g["spot"].first(),
    })


def detect_switches(
    store: ResearchHistoryStore,
    cal: ExchangeCalendar,
    instrument: str,
    source: str,
    oi_lo: float = 0.5,
    basis_jump_pct: float = 0.004,
) -> Dict:
    edges = _day_edges(store, instrument, source, SPOT_FOR[instrument])
    if edges.empty:
        return {"accepted": [], "disputed": [], "days": 0}
    days: List[date] = list(edges.index)
    expiries = set(monthly_expiries(cal, days[0], days[-1]))
    accepted, disputed = [], []
    for prev, cur in zip(days, days[1:]):
        p, c = edges.loc[prev], edges.loc[cur]
        votes = {
            "calendar": prev in expiries,
            "oi": bool(p["last_oi"] and c["first_oi"]
                       and not (oi_lo <= float(c["first_oi"]) / float(p["last_oi"]) <= 1.0 / oi_lo)),
            "basis": bool(abs(float(c["first_basis"]) - float(p["last_basis"])) > basis_jump_pct * float(c["spot"])),
        }
        n = sum(votes.values())
        if n == 0:
            continue
        rec = {
            "day": cur, "prev_day": prev, "votes": votes,
            "oi_ratio": round(float(c["first_oi"]) / float(p["last_oi"]), 3) if p["last_oi"] else None,
            "basis_change": round(float(c["first_basis"]) - float(p["last_basis"]), 1),
        }
        (accepted if n >= 2 else disputed).append(rec)
    return {"accepted": accepted, "disputed": disputed, "days": len(days)}


def apply_segments(store: ResearchHistoryStore, instrument: str, source: str, report: Dict) -> int:
    """Write segment ids (0..N) to every bar of the series from the ACCEPTED boundaries."""
    starts = sorted(b["day"] for b in report["accepted"])
    rows = store.conn.execute(
        "SELECT DISTINCT CAST(ts AS DATE) FROM hist_bars WHERE instrument = ? AND source = ? ORDER BY 1",
        [instrument, source],
    ).fetchall()
    mapping = []
    seg = 0
    for (d,) in rows:
        while seg < len(starts) and d >= starts[seg]:
            seg += 1
        mapping.append((d, seg))
    m = pd.DataFrame(mapping, columns=["day", "segment_id"])
    store.conn.register("segmap", m)
    try:
        store._write(
            """UPDATE hist_bars SET segment_id = segmap.segment_id FROM segmap
               WHERE hist_bars.instrument = ? AND hist_bars.source = ? AND CAST(hist_bars.ts AS DATE) = segmap.day""",
            [instrument, source],
        )
    finally:
        store.conn.unregister("segmap")
    return len(starts)


def segment_all(store: ResearchHistoryStore, cal: ExchangeCalendar) -> Dict[str, Dict]:
    """Detect and apply segments for every stitched futures series present. Returns reports."""
    reports = {}
    present = store.conn.execute(
        "SELECT DISTINCT instrument, source FROM hist_bars WHERE source IN ('CONT', 'CONT_NEXT')"
    ).fetchall()
    for inst, src in sorted(present):
        if inst not in SPOT_FOR:
            continue
        rep = detect_switches(store, cal, inst, src)
        apply_segments(store, inst, src, rep)
        reports[f"{inst}/{src}"] = rep
    return reports


__all__ = ["detect_switches", "apply_segments", "segment_all", "SPOT_FOR"]
