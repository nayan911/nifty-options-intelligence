"""Unit tests for TechnicalAnalysisEngine and SymbolTechnicalState.

Verifies:
1. Strict timeframe isolation (1m, 5m, 15m independent histories without cross-contamination).
2. Deterministic indicator warm-up requirements (no premature EMAs or fabricated ATRs).
3. Timestamp-synchronized cross-asset volume pairing (rejection of stale futures volume > 120s).
4. Breakout candidate state transitions and safety notification callback.
"""

from datetime import datetime, timedelta

import pytz

from nifty_engine.analysis.technical.breakouts import BreakoutContext, BreakoutDirection
from nifty_engine.analysis.technical.engine import (
    SymbolTechnicalState,
    TechnicalAnalysisEngine,
    TechnicalSnapshot,
)
from nifty_engine.core.interfaces import BreakoutState, Candle

IST = pytz.timezone("Asia/Kolkata")


def test_timeframe_isolation_prevents_contamination():
    """Verify that 1m, 5m, and 15m candles are strictly routed into separate deques
    and do not contaminate each other's indicator calculations.
    """
    state = SymbolTechnicalState(symbol="NSE:NIFTY26NOVFUT", is_index=False)
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Feed 10 1m candles
    for i in range(10):
        c1m = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24995.0,
            close=25005.0,
            volume=1000,
            is_closed=True,
        )
        snap = state.process_candle(c1m)
        assert snap.timeframe == "1m"

    # Feed two 5m candles (which have different prices and larger volumes)
    c5m_1 = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=base_time,
        open=25000.0,
        high=25050.0,
        low=24980.0,
        close=25040.0,
        volume=5000,
        is_closed=True,
    )
    c5m_2 = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=base_time + timedelta(minutes=5),
        open=25040.0,
        high=25060.0,
        low=25020.0,
        close=25055.0,
        volume=6000,
        is_closed=True,
    )
    snap_5m = state.process_candle(c5m_1)
    assert snap_5m.timeframe == "5m"
    state.process_candle(c5m_2)

    # Feed one 15m candle
    c15m = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="15m",
        timestamp=base_time,
        open=25000.0,
        high=25070.0,
        low=24970.0,
        close=25065.0,
        volume=15000,
        is_closed=True,
    )
    snap_15m = state.process_candle(c15m)
    assert snap_15m.timeframe == "15m"

    # Invariants: Strict separation of timeframe collections
    assert len(state.candles_1m) == 10
    assert len(state.candles_5m) == 2
    assert len(state.candles_15m) == 1

    # Verify that all candles in candles_1m have timeframe == '1m'
    assert all(c.timeframe == "1m" for c in state.candles_1m)
    assert all(c.timeframe == "5m" for c in state.candles_5m)
    assert all(c.timeframe == "15m" for c in state.candles_15m)


def test_indicator_warm_up_requirements_and_no_fabricated_atr():
    """Verify indicators remain None during documented warm-up periods
    and ATR is NEVER fabricated with a hardcoded constant.
    """
    state = SymbolTechnicalState(symbol="NSE:NIFTY26NOVFUT", is_index=False)
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # 1. Feed only 5 bars
    for i in range(5):
        c = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=1000,
            is_closed=True,
        )
        snap = state.process_candle(c)

    # Invariant: At 5 bars, indicators MUST NOT be prematurely available
    assert snap.is_warmed_up is False
    assert snap.ema_9 is None      # Requires >= 9
    assert snap.ema_21 is None     # Requires >= 21
    assert snap.ema_50 is None     # Requires >= 50
    assert snap.ema_200 is None    # Requires >= 200
    assert snap.rsi is None        # Requires >= 15
    assert snap.atr is None        # Strictly None! NOT 15.0!

    # 2. Feed bars 5 to 14 (total 15 bars)
    for i in range(5, 15):
        c = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=1000,
            is_closed=True,
        )
        snap = state.process_candle(c)

    # Invariant: At 15 bars, EMA9, RSI, and ATR become available, but not EMA21/50/200
    assert snap.ema_9 is not None
    assert snap.ema_21 is None
    assert snap.ema_50 is None
    assert snap.ema_200 is None
    assert snap.rsi is not None
    assert snap.atr is not None
    assert snap.is_warmed_up is False  # Requires >= 21 bars for full warm-up

    # 3. Feed bars 15 to 21 (total 22 bars)
    for i in range(15, 22):
        c = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=1000,
            is_closed=True,
        )
        snap = state.process_candle(c)

    assert snap.is_warmed_up is True
    assert snap.ema_21 is not None
    assert snap.ema_50 is None
    assert snap.ema_200 is None


def test_futures_rvol_freshness_synchronization():
    """Verify that spot index rejects stale futures volume (> 120s old)."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # 1. Warm up futures with 20 bars so RVOL is computed
    for i in range(20):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=i),
                open=25000.0,
                high=25020.0,
                low=24980.0,
                close=25010.0,
                volume=5000,
                is_closed=True,
            )
        )

    # Latest futures bar timestamp is 09:34
    fut_snap = engine.get_snapshot("NSE:NIFTY26NOVFUT")
    assert fut_snap.rvol is not None
    assert fut_snap.rvol > 0

    # 2. Spot index candle arriving at same time (09:34) -> FRESH (< 120s)
    spot_candle_fresh = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=19),  # 09:34
        open=25000.0,
        high=25020.0,
        low=24980.0,
        close=25010.0,
        volume=0,
        is_closed=True,
    )
    spot_snap_fresh = engine.on_candle(spot_candle_fresh)
    # Spot must receive fresh futures RVOL
    assert spot_snap_fresh.rvol == fut_snap.rvol

    # 3. Spot index candle arriving at 09:40 (6 minutes later, without new futures ticks) -> STALE (> 120s)
    spot_candle_stale = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=25),  # 09:40
        open=25010.0,
        high=25030.0,
        low=24990.0,
        close=25025.0,
        volume=0,
        is_closed=True,
    )
    spot_snap_stale = engine.on_candle(spot_candle_stale)
    # Invariant: Stale futures RVOL must be REJECTED (set to None)
    assert spot_snap_stale.rvol is None


def test_technical_engine_breakout_candidate_dispatch():
    """Verify that volume-confirmed breakout and retest dispatches to on_setup_candidate."""
    triggered_candidates = []

    def handle_candidate(setup: BreakoutContext, snap: TechnicalSnapshot):
        triggered_candidates.append((setup, snap))

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=handle_candidate,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    engine.set_daily_references(
        symbol="NSE:NIFTY26NOVFUT",
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Warm up with 22 bars below resistance (25080)
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=i),
                open=25075.0,
                high=25085.0,
                low=25070.0,
                close=25080.0,
                volume=1000,
                is_closed=True,
            )
        )

    # 2. Breakout candle with high volume surge (RVOL ~ 3.5x)
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=22),
            open=25085.0,
            high=25120.0,
            low=25080.0,
            close=25115.0,
            volume=3500,
            is_closed=True,
        )
    )

    # 3. Retest bar: price dips back to test 25100, closes green at 25108
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=23),
            open=25102.0,
            high=25112.0,
            low=25098.0,
            close=25108.0,
            volume=1200,
            is_closed=True,
        )
    )

    assert len(triggered_candidates) >= 1
    setup, snap = triggered_candidates[0]
    assert setup.current_state == BreakoutState.SETUP_ELIGIBLE
    assert setup.direction == BreakoutDirection.BULLISH
    assert snap.is_warmed_up is True
    assert setup.stop_loss is not None
    assert setup.target_price is not None


def test_futures_volume_lookahead_prevention():
    """Verify that a futures observation timestamped LATER than the spot candle
    is strictly rejected (no future lookahead allowed during replay).
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # 1. Warm up futures to 09:35
    for i in range(21):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=i),
                open=25000.0,
                high=25020.0,
                low=24980.0,
                close=25010.0,
                volume=5000,
                is_closed=True,
            )
        )

    # Futures timestamp is currently 09:35 (minutes=20)
    # 2. Simulate replaying a spot candle from 09:30 (5 minutes in the past of current futures state)
    past_spot_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=15),  # 09:30 < 09:35
        open=25000.0,
        high=25010.0,
        low=24990.0,
        close=25005.0,
        volume=0,
        is_closed=True,
    )
    snap = engine.on_candle(past_spot_candle)

    # Invariant: Look-ahead is prohibited! delta_seconds is negative, so rvol must be None!
    assert snap.rvol is None


def test_explicit_index_futures_mapping_isolation():
    """Verify that BANKNIFTY does NOT receive NIFTY futures volume and uses its own mapped contract."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    engine.set_index_futures_mapping("NSE:NIFTYBANK-INDEX", "NSE:BANKNIFTY26NOVFUT")
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Feed NIFTY futures with volume
    for i in range(20):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=i),
                open=25000.0,
                high=25010.0,
                low=24990.0,
                close=25005.0,
                volume=10000,
                is_closed=True,
            )
        )

    # Feed BANKNIFTY spot candle at same timestamp
    bnf_candle = Candle(
        symbol="NSE:NIFTYBANK-INDEX",
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=19),
        open=52000.0,
        high=52050.0,
        low=51980.0,
        close=52020.0,
        volume=0,
        is_closed=True,
    )
    bnf_snap = engine.on_candle(bnf_candle)

    # Invariant: BANKNIFTY has NO volume from NIFTY futures!
    assert bnf_snap.rvol is None

    # Feed BANKNIFTY futures with volume
    for i in range(20):
        engine.on_candle(
            Candle(
                symbol="NSE:BANKNIFTY26NOVFUT",
                timeframe="1m",
                timestamp=base_time + timedelta(minutes=i),
                open=52000.0,
                high=52050.0,
                low=51980.0,
                close=52030.0,
                volume=8000,
                is_closed=True,
            )
        )

    # Re-evaluate BANKNIFTY spot candle with fresh BANKNIFTY futures
    bnf_candle_2 = Candle(
        symbol="NSE:NIFTYBANK-INDEX",
        timeframe="1m",
        timestamp=base_time + timedelta(minutes=19),
        open=52000.0,
        high=52050.0,
        low=51980.0,
        close=52020.0,
        volume=0,
        is_closed=True,
    )
    bnf_snap_2 = engine.on_candle(bnf_candle_2)
    assert bnf_snap_2.rvol is not None


def test_multi_timeframe_snapshots_retained():
    """Verify that get_snapshot retains independent 1m, 5m, and 15m snapshots."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    c1m = Candle(symbol=sym, timeframe="1m", timestamp=base_time, open=25000, high=25010, low=24990, close=25005, volume=1000, is_closed=True)
    c5m = Candle(symbol=sym, timeframe="5m", timestamp=base_time, open=25000, high=25050, low=24980, close=25040, volume=5000, is_closed=True)
    c15m = Candle(symbol=sym, timeframe="15m", timestamp=base_time, open=25000, high=25080, low=24970, close=25070, volume=15000, is_closed=True)

    engine.on_candle(c1m)
    engine.on_candle(c5m)
    engine.on_candle(c15m)

    snap_1m = engine.get_snapshot(sym, timeframe="1m")
    snap_5m = engine.get_snapshot(sym, timeframe="5m")
    snap_15m = engine.get_snapshot(sym, timeframe="15m")

    assert snap_1m is not None and snap_1m.timeframe == "1m" and snap_1m.close == 25005.0
    assert snap_5m is not None and snap_5m.timeframe == "5m" and snap_5m.close == 25040.0
    assert snap_15m is not None and snap_15m.timeframe == "15m" and snap_15m.close == 25070.0
