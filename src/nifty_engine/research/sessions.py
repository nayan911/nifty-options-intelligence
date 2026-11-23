"""Exchange-calendar-derived expectations for research data.

`expected_minutes` gives the exact 1m bar start times a series should have on a day, so data
completeness is checked minute by minute instead of with a fixed bar-count threshold:
- CASH (indices, equities, India VIX): 09:15..15:29 -> 375 bars on a normal session.
- FNO (futures): 09:15 .. (F&O close - 1 min). The close moved from 15:30 to 15:40 on
  31-Jul-2026 (verified in FYERS history), so this is 375 bars before and 385 bars after.
- Special sessions (Budget Sunday, Muhurat) use their configured window.
Bars outside these windows (e.g. pre-open 09:08-09:14 prints seen on 7-9 Sep 2026) are
out-of-session and must never feed features.
"""

from datetime import date, datetime, timedelta
from typing import List

import pytz

from nifty_engine.core.calendar import ExchangeCalendar

IST = pytz.timezone("Asia/Kolkata")

CASH = "CASH"
FNO = "FNO"


def session_window(cal: ExchangeCalendar, day: date, kind: str):
    """(start, end_exclusive) times of the trading window for `kind` on `day`, or None if closed."""
    if not cal.is_trading_day(day):
        return None
    if day in cal.special_sessions:
        cfg = cal.special_sessions[day]
        start = datetime.strptime(cfg["session_start"], "%H:%M:%S").time()
        cash_end = datetime.strptime(cfg["session_end"], "%H:%M:%S").time()
        end = cash_end
        if kind == FNO and cfg.get("derivatives_close"):
            end = datetime.strptime(cfg["derivatives_close"], "%H:%M:%S").time()
        return start, end
    end = cal.cash_close if kind == CASH else cal.derivatives_close_on(day)
    return cal.market_open, end


def expected_minutes(cal: ExchangeCalendar, day: date, kind: str) -> List[datetime]:
    window = session_window(cal, day, kind)
    if window is None:
        return []
    start, end = window
    cur = IST.localize(datetime.combine(day, start))
    stop = IST.localize(datetime.combine(day, end))
    out = []
    while cur < stop:
        out.append(cur)
        cur += timedelta(minutes=1)
    return out


def monthly_expiries(cal: ExchangeCalendar, start: date, end: date, weekday: int = 1) -> List[date]:
    """Monthly index-derivative expiries between start and end: the last `weekday` (Tuesday) of each
    month, moved to the previous trading day when that day is a holiday (e.g. 31-Mar-2026 -> 30-Mar)."""
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        nxt = date(y + (m == 12), (m % 12) + 1, 1)
        d = nxt - timedelta(days=1)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        while not cal.is_trading_day(d):
            d -= timedelta(days=1)
        if start <= d <= end:
            out.append(d)
        y, m = nxt.year, nxt.month
    return out


def kind_for(instrument_kind: str) -> str:
    """Map a SeriesSpec kind (SPOT/CONT/OWN) to a session kind."""
    return FNO if instrument_kind in ("CONT", "CONT_NEXT", "OWN") else CASH


def minute_floor(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


__all__ = ["CASH", "FNO", "expected_minutes", "monthly_expiries", "session_window", "kind_for", "minute_floor"]
