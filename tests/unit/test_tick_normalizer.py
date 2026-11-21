"""Unit tests for TickNormalizer and data sanitization."""

from datetime import datetime, timedelta

import pytz

from nifty_engine.core.interfaces import Tick
from nifty_engine.data.tick_normalizer import TickNormalizer, TickStatus

IST = pytz.timezone("Asia/Kolkata")


def test_futures_incremental_volume_and_baseline():
    normalizer = TickNormalizer()
    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)
    sym = "NSE:NIFTY26NOVFUT"

    # Tick 1: Baseline established
    t1 = Tick(symbol=sym, exchange_timestamp=base_time, ltp=24100.0, cumulative_volume=10000)
    res1 = normalizer.normalize(t1, current_time=base_time)
    assert res1.status == TickStatus.VALID_LIVE
    assert res1.tick.incremental_volume == 0
    assert res1.tick.cumulative_volume == 10000

    # Tick 2: Volume increased to 10250 -> Delta = 250
    t2_time = base_time + timedelta(seconds=1)
    t2 = Tick(symbol=sym, exchange_timestamp=t2_time, ltp=24102.0, cumulative_volume=10250)
    res2 = normalizer.normalize(t2, current_time=t2_time)
    assert res2.status == TickStatus.VALID_LIVE
    assert res2.tick.incremental_volume == 250
    assert res2.tick.cumulative_volume == 10250


def test_spot_index_volume_strictly_zero():
    normalizer = TickNormalizer()
    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)
    t = Tick(symbol="NSE:NIFTY50-INDEX", exchange_timestamp=base_time, ltp=24100.0, cumulative_volume=999999)
    res = normalizer.normalize(t, current_time=base_time)
    assert res.tick.is_index is True
    assert res.tick.incremental_volume == 0
    assert res.tick.cumulative_volume == 0


def test_out_of_order_tick_quarantine_prevents_baseline_corruption():
    normalizer = TickNormalizer()
    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)
    sym = "NSE:NIFTY26NOVFUT"

    # Tick 1 at 09:15:05, cum_vol=10000
    t1_time = base_time + timedelta(seconds=5)
    t1 = Tick(symbol=sym, exchange_timestamp=t1_time, ltp=24100.0, cumulative_volume=10000)
    normalizer.normalize(t1, current_time=t1_time)

    # Tick 2 at 09:15:10, cum_vol=10500
    t2_time = base_time + timedelta(seconds=10)
    t2 = Tick(symbol=sym, exchange_timestamp=t2_time, ltp=24105.0, cumulative_volume=10500)
    res2 = normalizer.normalize(t2, current_time=t2_time)
    assert res2.tick.incremental_volume == 500

    # Tick 3: Out-of-order back-dated packet arriving with time 09:15:02 and cum_vol=9800
    t3 = Tick(symbol=sym, exchange_timestamp=base_time + timedelta(seconds=2), ltp=24095.0, cumulative_volume=9800)
    res3 = normalizer.normalize(t3, current_time=t2_time)
    assert res3.status == TickStatus.OUT_OF_ORDER

    # Tick 4 at 09:15:15 with cum_vol=10800
    # Must compute delta against 10500 (tick 2), NOT against corrupted 9800!
    t4_time = base_time + timedelta(seconds=15)
    t4 = Tick(symbol=sym, exchange_timestamp=t4_time, ltp=24110.0, cumulative_volume=10800)
    res4 = normalizer.normalize(t4, current_time=t4_time)
    assert res4.status == TickStatus.VALID_LIVE
    assert res4.tick.incremental_volume == 300  # 10800 - 10500


def test_duplicate_tick_rejection():
    normalizer = TickNormalizer()
    base_time = datetime(2026, 11, 23, 9, 15, 0, tzinfo=IST)
    sym = "NSE:NIFTY26NOVFUT"

    t1 = Tick(symbol=sym, exchange_timestamp=base_time, ltp=24100.0, cumulative_volume=10000)
    res1 = normalizer.normalize(t1, current_time=base_time)
    assert res1.status == TickStatus.VALID_LIVE

    # Duplicate tick (same timestamp, ltp, cumulative_vol)
    res2 = normalizer.normalize(t1, current_time=base_time)
    assert res2.status == TickStatus.DUPLICATE


def test_holiday_snapshot_classified():
    normalizer = TickNormalizer()
    # Saturday Nov 21, 2026 (Holiday/Weekend)
    sat_time = datetime(2026, 11, 21, 14, 0, 0, tzinfo=IST)
    t = Tick(symbol="NSE:NIFTY26NOVFUT", exchange_timestamp=sat_time, ltp=24100.0, cumulative_volume=50000)
    res = normalizer.normalize(t, current_time=sat_time)
    assert res.status == TickStatus.VALID_SNAPSHOT


def test_past_trading_day_timestamp_on_weekend_classified_as_snapshot():
    """Regression test:
    FYERS sends Friday 14:00 IST quote during Saturday weekend session.
    Because current time is Saturday (market closed), quote must be VALID_SNAPSHOT, NOT VALID_LIVE.
    """
    normalizer = TickNormalizer()
    fri_quote_time = datetime(2026, 11, 20, 14, 0, 0, tzinfo=IST)
    sat_wall_time = datetime(2026, 11, 21, 12, 0, 0, tzinfo=IST)

    t = Tick(symbol="NSE:NIFTY26NOVFUT", exchange_timestamp=fri_quote_time, ltp=24100.0, cumulative_volume=50000)
    res = normalizer.normalize(t, current_time=sat_wall_time)
    assert res.status == TickStatus.VALID_SNAPSHOT


def test_stale_quote_during_open_market_classified_as_snapshot():
    """Regression test:
    Market is open on Monday at 10:00 IST, but broker sends a stale Friday quote.
    Must be classified as VALID_SNAPSHOT, never VALID_LIVE.
    """
    normalizer = TickNormalizer(max_freshness_seconds=30.0)
    mon_wall_time = datetime(2026, 11, 23, 10, 0, 0, tzinfo=IST)
    fri_quote_time = datetime(2026, 11, 20, 15, 29, 0, tzinfo=IST)

    t = Tick(symbol="NSE:NIFTY26NOVFUT", exchange_timestamp=fri_quote_time, ltp=24100.0, cumulative_volume=50000)
    res = normalizer.normalize(t, current_time=mon_wall_time)
    assert res.status == TickStatus.VALID_SNAPSHOT


def test_stale_latency_during_open_market():
    """Quote is from today (Monday), but 5 minutes old (> 30s threshold)."""
    normalizer = TickNormalizer(max_freshness_seconds=30.0)
    mon_wall_time = datetime(2026, 11, 23, 10, 5, 0, tzinfo=IST)
    mon_old_quote = datetime(2026, 11, 23, 10, 0, 0, tzinfo=IST)

    t = Tick(symbol="NSE:NIFTY26NOVFUT", exchange_timestamp=mon_old_quote, ltp=24100.0, cumulative_volume=50000)
    res = normalizer.normalize(t, current_time=mon_wall_time)
    assert res.status == TickStatus.VALID_SNAPSHOT
