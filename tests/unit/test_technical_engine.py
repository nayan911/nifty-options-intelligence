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


def test_15m_market_structure_provisional_and_confirmed():
    """Verify 15-minute market structure analysis:
    1. Early session returns is_provisional=True with structural level trend bias.
    2. After 21 candles, indicator warm-up completes and confirmed trend bias is evaluated.
    3. 15m Opening Range, daily reference levels, and S/R zones are computed.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY50-INDEX"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=24950.0,
        cdo=25000.0,
    )

    # First 15m candle (09:15-09:30): opens at CDO (25000), rallies to 25050, closes at 25040
    c1 = Candle(
        symbol=sym,
        timeframe="15m",
        timestamp=base_time,
        open=25000.0,
        high=25050.0,
        low=24980.0,
        close=25040.0,
        volume=0,
        is_closed=True,
    )
    snap1 = engine.on_candle(c1)
    assert snap1.timeframe == "15m"
    assert snap1.is_warmed_up is False
    assert snap1.is_provisional is True
    # Price (25040) > CDO (25000) and > PDC (24950) -> provisional BULLISH
    assert snap1.trend_bias == "BULLISH"
    assert snap1.nearest_support is not None or snap1.nearest_resistance is not None

    # Feed 21 more 15m candles in a steady uptrend to warm up EMA 21
    for i in range(1, 22):
        c = Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=base_time + timedelta(minutes=15 * i),
            open=25040.0 + (i * 10),
            high=25055.0 + (i * 10),
            low=25035.0 + (i * 10),
            close=25050.0 + (i * 10),
            volume=0,
            is_closed=True,
        )
        snap = engine.on_candle(c)

    assert snap.is_warmed_up is True
    assert snap.is_provisional is False
    assert snap.ema_9 is not None
    assert snap.ema_21 is not None
    assert snap.rsi is not None
    assert snap.atr is not None
    assert snap.ema_9 > snap.ema_21
    assert snap.trend_bias == "BULLISH"


def test_5m_setup_analysis_indicators_and_vwap():
    """Verify 5-minute setup analysis:
    1. EMA 9/21/50, RSI 14, ATR 14, and futures VWAP calculations.
    2. 5m RVOL computation from 5m volume history.
    3. Retains 5m snapshot.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    # Feed 22 5m candles with realistic prices and volumes
    for i in range(22):
        c = Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * i),
            open=25000.0 + i * 5,
            high=25015.0 + i * 5,
            low=24995.0 + i * 5,
            close=25010.0 + i * 5,
            volume=5000 + i * 200,
            is_closed=True,
        )
        snap = engine.on_candle(c)

    assert snap.timeframe == "5m"
    assert snap.is_warmed_up is True
    assert snap.ema_9 is not None
    assert snap.ema_21 is not None
    assert snap.rsi is not None
    assert snap.atr is not None
    assert snap.vwap is not None
    assert snap.vwap_upper is not None
    assert snap.vwap_lower is not None
    assert snap.rvol is not None

    state = engine.states[sym]
    assert state.latest_5m_rvol is not None


def test_5m_spot_index_inherits_5m_futures_rvol():
    """Verify cross-asset pairing on the 5-minute timeframe:
    Spot index inherits 5m futures RVOL if within 300s window.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Feed 22 5m candles to futures
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25000.0,
                high=25015.0,
                low=24995.0,
                close=25010.0,
                volume=10000,
                is_closed=True,
            )
        )

    # 1. Spot candle at same 5m timestamp (delta = 0) inherits 5m futures RVOL
    spot_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=base_time + timedelta(minutes=5 * 21),
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25015.0,
        volume=0,
        is_closed=True,
    )
    snap = engine.on_candle(spot_candle)
    assert snap.rvol is not None

    # 2. Spot candle timestamped > 300s after latest futures does NOT inherit stale RVOL
    stale_spot_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=base_time + timedelta(minutes=5 * 21 + 6),  # 360 seconds later
        open=25015.0,
        high=25025.0,
        low=25000.0,
        close=25020.0,
        volume=0,
        is_closed=True,
    )
    stale_snap = engine.on_candle(stale_spot_candle)
    assert stale_snap.rvol is None


def test_5m_breakout_state_machine_isolation_and_callback():
    """Verify that 5m breakout state machine:
    1. Operates independently from 1m breakout state machine.
    2. Detects breakout and retest across 5m resistance.
    3. Triggers candidate notification callback with timeframe="5m".
    """
    candidates = []

    def on_cand(setup: BreakoutContext, snap: TechnicalSnapshot):
        candidates.append((setup, snap))

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=on_cand,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Warm up with 22 5m bars below resistance (25080)
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=2000,
                is_closed=True,
            )
        )

    # 2. 5m Breakout candle with high volume surge (RVOL > 1.3x)
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * 22),
            open=25085.0,
            high=25125.0,
            low=25080.0,
            close=25120.0,
            volume=8000,
            is_closed=True,
        )
    )

    # 3. 5m Retest bar: price dips to test 25100, closes green at 25110
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * 23),
            open=25105.0,
            high=25115.0,
            low=25098.0,
            close=25110.0,
            volume=3000,
            is_closed=True,
        )
    )

    assert len(candidates) >= 1
    setup, snap = candidates[0]
    assert setup.current_state == BreakoutState.SETUP_ELIGIBLE
    assert setup.direction == BreakoutDirection.BULLISH
    assert snap.timeframe == "5m"
    assert setup.stop_loss is not None
    assert setup.target_price is not None

    # Invariant: 1m active breakouts must be completely empty/unaffected!
    state = engine.states[sym]
    assert len(state.active_breakouts_1m) == 0


def test_breakout_candidate_notification_exactly_once():
    """Verify Issue P1 Fix: Setup notifications are emitted exactly once upon initial transition to SETUP_ELIGIBLE.
    Subsequent candles on the same level do NOT trigger duplicate callbacks.
    """
    callback_calls = []

    def on_candidate(setup: BreakoutContext, snap: TechnicalSnapshot):
        callback_calls.append((setup.setup_id, snap.timestamp))

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=on_candidate,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Warm up with 22 5m candles below resistance
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=2000,
                is_closed=True,
            )
        )

    # 2. Breakout candle
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * 22),
            open=25085.0,
            high=25125.0,
            low=25080.0,
            close=25120.0,
            volume=8000,
            is_closed=True,
        )
    )

    # 3. Retest bar -> transitions to SETUP_ELIGIBLE
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * 23),
            open=25105.0,
            high=25115.0,
            low=25098.0,
            close=25110.0,
            volume=3000,
            is_closed=True,
        )
    )

    initial_count = len(callback_calls)
    assert initial_count >= 1
    # Verify all emitted setups at transition have unique setup IDs
    assert len(set(c[0] for c in callback_calls)) == initial_count

    # 4. Feed subsequent candles while price stays above resistance
    for k in range(24, 27):
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * k),
                open=25112.0,
                high=25130.0,
                low=25105.0,
                close=25125.0,
                volume=2500,
                is_closed=True,
            )
        )

    # Invariant: Callback was called EXACTLY ONCE per setup, never re-triggered on subsequent candles!
    assert len(callback_calls) == initial_count


def test_opening_range_session_date_isolation():
    """Verify Issue P1 Fix: Opening Range is isolated strictly to the current session date.
    Prior trading day candles do not contaminate the new session's ORH/ORL.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    friday_time = IST.localize(datetime(2026, 11, 20, 9, 15))
    monday_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    # Friday 15m candle: High 25500, Low 25300
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=friday_time,
            open=25400.0,
            high=25500.0,
            low=25300.0,
            close=25450.0,
            volume=5000,
            is_closed=True,
        )
    )

    # Monday 15m candle: High 24200, Low 24000
    snap_monday = engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=monday_time,
            open=24100.0,
            high=24200.0,
            low=24000.0,
            close=24150.0,
            volume=4000,
            is_closed=True,
        )
    )
    assert snap_monday.timeframe == "15m"

    # Nearest S/R zones on Monday must reflect Monday's Opening Range, NOT Friday's 25500/25300!
    state = engine.states[sym]
    assert state.current_session_date == monday_time.date()
    zones = state.latest_snapshots["15m"]
    assert zones is not None
    if zones.nearest_resistance:
        assert zones.nearest_resistance.nominal_price < 25000.0


def test_breakout_setup_expires_when_dynamic_zone_shifts():
    """Verify Issue P1 Fix: Setups retain their original reference zone snapshot.
    If dynamic levels (VWAP or swing clusters) shift away, the setup still counts bars and expires.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY26NOVFUT"

    engine.set_daily_references(
        symbol=sym,
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Warm up 22 5m candles below resistance
    for i in range(22):
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=2000,
                is_closed=True,
            )
        )

    # 2. Breakout candle creates in-flight setup on resistance (25100)
    engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * 22),
            open=25085.0,
            high=25125.0,
            low=25080.0,
            close=25120.0,
            volume=8000,
            is_closed=True,
        )
    )

    # 3. Price runs away without retesting (5 bars later > max_retest_candles=4)
    for j in range(23, 28):
        engine.on_candle(
            Candle(
                symbol=sym,
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * j),
                open=25140.0 + (j - 23) * 10,
                high=25160.0 + (j - 23) * 10,
                low=25135.0 + (j - 23) * 10,
                close=25150.0 + (j - 23) * 10,
                volume=2000,
                is_closed=True,
            )
        )

    # Invariant: Active breakout setup on PDH_25100.0 must NOT linger indefinitely; it has expired and been removed!
    state = engine.states[sym]
    assert "PDH_25100.0" not in state.active_breakouts_5m


def test_provisional_trend_bias_incorporates_opening_range_without_daily_levels():
    """Verify Issue P2 Fix: Early morning session establishes provisional trend bias
    using the completed 15m Opening Range structure without requiring manual daily reference levels.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    sym = "NSE:NIFTY50-INDEX"

    # Notice: NO call to engine.set_daily_references()!

    # 1. 09:15-09:30 15m candle forms Opening Range: High 25050, Low 24950, Close 25020 (Open 25000)
    snap1 = engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=base_time,
            open=25000.0,
            high=25050.0,
            low=24950.0,
            close=25020.0,
            volume=0,
            is_closed=True,
        )
    )
    # Price is above open price (25000) -> BULLISH (provisional)
    assert snap1.is_provisional is True
    assert snap1.trend_bias == "BULLISH"

    # 2. 09:30-09:45 candle breaks out above Opening Range High (25050) to 25070
    snap2 = engine.on_candle(
        Candle(
            symbol=sym,
            timeframe="15m",
            timestamp=base_time + timedelta(minutes=15),
            open=25025.0,
            high=25075.0,
            low=25020.0,
            close=25070.0,
            volume=0,
            is_closed=True,
        )
    )
    assert snap2.trend_bias == "BULLISH"
    assert snap2.is_provisional is True


def test_spot_and_futures_deterministic_arrival_order_independence():
    """Verify Issue P2 Fix: Spot index technical evaluation is deterministic
    regardless of whether the spot candle arrives before or after its paired futures candle.
    """
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))
    t21 = base_time + timedelta(minutes=5 * 21)

    # Engine A: Futures arrives first, Spot arrives second
    engine_a = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    for i in range(22):
        engine_a.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25000.0,
                high=25015.0,
                low=24995.0,
                close=25010.0,
                volume=10000,
                is_closed=True,
            )
        )
    spot_candle_a = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=t21,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25015.0,
        volume=0,
        is_closed=True,
    )
    snap_a = engine_a.on_candle(spot_candle_a)

    # Engine B: Spot arrives first, Futures arrives second
    engine_b = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    # Warm up futures up to bar 20
    for i in range(21):
        engine_b.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=base_time + timedelta(minutes=5 * i),
                open=25000.0,
                high=25015.0,
                low=24995.0,
                close=25010.0,
                volume=10000,
                is_closed=True,
            )
        )
    # Feed spot candle at t21 FIRST
    spot_candle_b = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=t21,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25015.0,
        volume=0,
        is_closed=True,
    )
    engine_b.on_candle(spot_candle_b)

    # Feed futures candle at t21 SECOND
    engine_b.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="5m",
            timestamp=t21,
            open=25000.0,
            high=25015.0,
            low=24995.0,
            close=25010.0,
            volume=10000,
            is_closed=True,
        )
    )
    snap_b = engine_b.get_snapshot("NSE:NIFTY50-INDEX", timeframe="5m")

    # Invariant: Both arrival orders yield identical RVOL on spot snapshot!
    assert snap_a is not None and snap_a.rvol is not None
    assert snap_b is not None and snap_b.rvol is not None
    assert snap_a.rvol == snap_b.rvol


def test_spot_breakout_arrival_order_parity_and_candidate_dispatch():
    """Verify Issue P0 Fix: Breakout state machine evaluation and candidate callbacks
    produce identical setup_id, state transitions, and parameters whether spot arrives first or futures arrives first.
    """
    candidates_a = []
    candidates_b = []

    def cb_a(setup: BreakoutContext, snap: TechnicalSnapshot):
        candidates_a.append((setup, snap))

    def cb_b(setup: BreakoutContext, snap: TechnicalSnapshot):
        candidates_b.append((setup, snap))

    engine_a = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=cb_a,
    )
    engine_b = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=cb_b,
    )

    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    for eng in (engine_a, engine_b):
        eng.set_daily_references(
            symbol="NSE:NIFTY50-INDEX",
            pdh=25100.0,
            pdl=24900.0,
            pdc=25000.0,
            cdo=25020.0,
        )

    # Warm up bars 0..21 (22 bars) on both spot and futures
    for i in range(22):
        ts = base_time + timedelta(minutes=5 * i)
        fut_c = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="5m",
            timestamp=ts,
            open=25070.0,
            high=25085.0,
            low=25065.0,
            close=25080.0,
            volume=2000,
            is_closed=True,
        )
        spot_c = Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="5m",
            timestamp=ts,
            open=25070.0,
            high=25085.0,
            low=25065.0,
            close=25080.0,
            volume=0,
            is_closed=True,
        )
        engine_a.on_candle(fut_c)
        engine_a.on_candle(spot_c)
        engine_b.on_candle(spot_c)
        engine_b.on_candle(fut_c)

    # Bar 22: Breakout bar (close 25120 above PDH 25100) with volume surge (8000 vs 2000 baseline)
    ts_breakout = base_time + timedelta(minutes=5 * 22)
    fut_bo = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=ts_breakout,
        open=25085.0,
        high=25125.0,
        low=25080.0,
        close=25120.0,
        volume=8000,
        is_closed=True,
    )
    spot_bo = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=ts_breakout,
        open=25085.0,
        high=25125.0,
        low=25080.0,
        close=25120.0,
        volume=0,
        is_closed=True,
    )

    # In Engine A: Futures arrives first, then spot
    engine_a.on_candle(fut_bo)
    engine_a.on_candle(spot_bo)

    # In Engine B: Spot arrives first, then futures
    engine_b.on_candle(spot_bo)
    engine_b.on_candle(fut_bo)

    # Bar 23: Retest bar (dips to 25098, closes 25110)
    ts_retest = base_time + timedelta(minutes=5 * 23)
    fut_re = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=ts_retest,
        open=25105.0,
        high=25115.0,
        low=25098.0,
        close=25110.0,
        volume=3000,
        is_closed=True,
    )
    spot_re = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=ts_retest,
        open=25105.0,
        high=25115.0,
        low=25098.0,
        close=25110.0,
        volume=0,
        is_closed=True,
    )

    # In Engine A: Futures first
    engine_a.on_candle(fut_re)
    engine_a.on_candle(spot_re)

    # In Engine B: Spot first
    engine_b.on_candle(spot_re)
    engine_b.on_candle(fut_re)

    # Invariant 1: Candidate callback dispatched in both engines
    spot_cand_a = [c for c in candidates_a if c[0].symbol == "NSE:NIFTY50-INDEX"]
    spot_cand_b = [c for c in candidates_b if c[0].symbol == "NSE:NIFTY50-INDEX"]

    assert len(spot_cand_a) >= 1
    assert len(spot_cand_b) >= 1
    assert len(spot_cand_a) == len(spot_cand_b)

    setup_a, snap_a = spot_cand_a[0]
    setup_b, snap_b = spot_cand_b[0]

    # Invariant 2: Candidate setup parameters are perfectly identical
    assert setup_a.setup_id == setup_b.setup_id
    assert setup_a.current_state == BreakoutState.SETUP_ELIGIBLE
    assert setup_b.current_state == BreakoutState.SETUP_ELIGIBLE
    assert setup_a.stop_loss == setup_b.stop_loss
    assert setup_a.target_price == setup_b.target_price
    assert setup_a.timeframe == setup_b.timeframe == "5m"
    assert setup_a.symbol == setup_b.symbol == "NSE:NIFTY50-INDEX"


def test_unclosed_and_duplicate_candle_validation():
    """Verify Issue P1 Fix:
    1. Unclosed candles (is_closed=False) are rejected and return None.
    2. Duplicate candles with same timestamp are rejected and do not double-advance state.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # 1. Unclosed candle rejected
    unclosed_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time,
        open=25000.0,
        high=25010.0,
        low=24990.0,
        close=25005.0,
        volume=0,
        is_closed=False,
    )
    assert engine.on_candle(unclosed_candle) is None
    assert len(engine.states["NSE:NIFTY50-INDEX"].candles_1m) == 0

    # 2. Closed candle accepted
    closed_c1 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time,
        open=25000.0,
        high=25010.0,
        low=24990.0,
        close=25005.0,
        volume=0,
        is_closed=True,
    )
    snap1 = engine.on_candle(closed_c1)
    assert snap1 is not None
    assert len(engine.states["NSE:NIFTY50-INDEX"].candles_1m) == 1

    # 3. Duplicate candle with same timestamp does NOT append new candle
    duplicate_c1 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time,
        open=25000.0,
        high=25015.0,
        low=24985.0,
        close=25010.0,
        volume=0,
        is_closed=True,
    )
    snap_dup = engine.on_candle(duplicate_c1)
    assert snap_dup is not None
    assert len(engine.states["NSE:NIFTY50-INDEX"].candles_1m) == 1


def test_setup_id_timeframe_isolation():
    """Verify Issue P2 Fix: Setup IDs include symbol and timeframe, preventing collision
    between 1m and 5m setups on the same reference level.
    """
    ctx_1m = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        breakout_candle_index=5,
        breakout_time=IST.localize(datetime(2026, 11, 23, 9, 20)),
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
    )
    ctx_5m = BreakoutContext(
        level_id="PDH_25100.0",
        level_type="PDH",
        nominal_price=25100.0,
        direction=BreakoutDirection.BULLISH,
        breakout_candle_index=1,
        breakout_time=IST.localize(datetime(2026, 11, 23, 9, 20)),
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
    )

    assert "1m" in ctx_1m.setup_id
    assert "5m" in ctx_5m.setup_id
    assert ctx_1m.setup_id != ctx_5m.setup_id


def test_session_rollover_clears_levels_and_snapshots():
    """Verify Issue P2 Fix: Session rollover resets daily reference levels, snapshots,
    notified setups, and pending evaluations so prior day state never leaks into new session.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    day1_time = IST.localize(datetime(2026, 11, 20, 15, 25))
    day2_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    engine.set_daily_references(
        symbol="NSE:NIFTY50-INDEX",
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # Day 1 candle
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="5m",
            timestamp=day1_time,
            open=25050.0,
            high=25070.0,
            low=25040.0,
            close=25060.0,
            volume=0,
            is_closed=True,
        )
    )

    state = engine.states["NSE:NIFTY50-INDEX"]
    assert state.daily_reference_levels is not None
    assert state.latest_snapshots["5m"] is not None

    # Day 2 candle triggers rollover
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="5m",
            timestamp=day2_time,
            open=25200.0,
            high=25220.0,
            low=25190.0,
            close=25210.0,
            volume=0,
            is_closed=True,
        )
    )

    # Rollover verified
    assert state.current_session_date == day2_time.date()
    assert state.daily_reference_levels is None
    assert len(state.notified_setup_ids) == 0
    assert len(state._pending_breakout_eval) == 0


def test_missing_futures_candle_followed_by_future_interval_candle():
    """Verify Issue P0 Fix: Future-interval volume CANNOT confirm a previous breakout.
    Scenario:
    1. 10:00 spot candle breaks out above resistance and waits for 10:00 futures confirmation.
    2. 10:00 futures candle is missing.
    3. 10:05 futures candle arrives with huge surge volume (RVOL = 2.5x).
    4. Invariant: 10:00 spot setup MUST NOT borrow the 10:05 futures RVOL.
       It completes with missing volume (futures_rvol=None), preventing look-ahead bias!
    """
    candidates = []

    def on_cand(setup: BreakoutContext, snap: TechnicalSnapshot):
        candidates.append((setup, snap))

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_candidate=on_cand,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    engine.set_daily_references(
        symbol="NSE:NIFTY50-INDEX",
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Warm up 22 bars (09:15 to 11:00) below resistance (25080)
    for i in range(22):
        ts = base_time + timedelta(minutes=5 * i)
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=ts,
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=2000,
                is_closed=True,
            )
        )
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY50-INDEX",
                timeframe="5m",
                timestamp=ts,
                open=25070.0,
                high=25085.0,
                low=25065.0,
                close=25080.0,
                volume=0,
                is_closed=True,
            )
        )

    # 2. 11:05 spot candle breaks out above PDH (close 25120 > 25100)
    ts_1105 = base_time + timedelta(minutes=5 * 22)
    spot_1105 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=ts_1105,
        open=25085.0,
        high=25125.0,
        low=25080.0,
        close=25120.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(spot_1105)

    spot_state = engine.states["NSE:NIFTY50-INDEX"]
    # Spot is pending futures confirmation for interval 11:05
    assert "5m" in spot_state._pending_breakout_eval
    pending_candle = spot_state._pending_breakout_eval["5m"][0]
    assert pending_candle.timestamp == ts_1105

    # 3. 11:05 futures candle is MISSING!
    # Next futures candle arrives for interval 11:10 (future interval) with massive volume (8000)
    ts_1110 = base_time + timedelta(minutes=5 * 23)
    fut_1110 = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=ts_1110,
        open=25085.0,
        high=25130.0,
        low=25080.0,
        close=25125.0,
        volume=8000,  # 4x volume surge!
        is_closed=True,
    )
    engine.on_candle(fut_1110)

    # Invariant: 11:05 spot pending breakout was flushed WITHOUT borrowing 11:10 futures volume
    assert "5m" not in spot_state._pending_breakout_eval
    # 11:05 spot snapshot must NOT have 11:10 futures RVOL
    snap_1105 = spot_state.latest_snapshots["5m"]
    assert snap_1105 is not None
    assert snap_1105.rvol is None  # Missing volume status maintained!


def test_duplicate_and_delayed_futures_observations():
    """Verify Issue P0 Fix: Delayed older futures observations do not confirm newer spot candles,
    and duplicate futures observations do not corrupt snapshot RVOL.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Warm up futures up to bar 21
    for i in range(22):
        ts = base_time + timedelta(minutes=5 * i)
        engine.on_candle(
            Candle(
                symbol="NSE:NIFTY26NOVFUT",
                timeframe="5m",
                timestamp=ts,
                open=25000.0,
                high=25015.0,
                low=24995.0,
                close=25010.0,
                volume=10000,
                is_closed=True,
            )
        )

    # Spot candle at interval 22 (11:05)
    t22 = base_time + timedelta(minutes=5 * 22)
    spot_c = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=t22,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25015.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(spot_c)

    spot_state = engine.states["NSE:NIFTY50-INDEX"]
    assert "5m" in spot_state._pending_breakout_eval

    # Feed an older delayed futures candle (e.g. at interval 20)
    t20 = base_time + timedelta(minutes=5 * 20)
    old_fut = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="5m",
        timestamp=t20,
        open=25000.0,
        high=25015.0,
        low=24995.0,
        close=25010.0,
        volume=5000,
        is_closed=True,
    )
    engine.on_candle(old_fut)

    # Invariant: Older futures candle did NOT confirm the newer t22 spot candle!
    assert "5m" in spot_state._pending_breakout_eval
    assert spot_state._pending_breakout_eval["5m"][0].timestamp == t22


def test_pending_evaluation_clock_driven_expiry_when_feed_stops():
    """Verify Issue P1 Fix: When feeds stop completely, clock-driven expiry clears pending
    evaluations and marks them with explicit missing volume rather than waiting indefinitely.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Feed 1 futures candle to initialize futures stream
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="5m",
            timestamp=base_time,
            open=25000.0,
            high=25015.0,
            low=24995.0,
            close=25010.0,
            volume=5000,
            is_closed=True,
        )
    )

    # Spot candle at 09:20 arrives and waits for futures volume
    ts_0920 = base_time + timedelta(minutes=5)
    spot_0920 = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=ts_0920,
        open=25000.0,
        high=25020.0,
        low=24990.0,
        close=25015.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(spot_0920)

    spot_state = engine.states["NSE:NIFTY50-INDEX"]
    assert "5m" in spot_state._pending_breakout_eval

    # Feed stops completely! Clock advances by 15 minutes to 09:35 (past 5m bar close + timeout)
    clock_now = ts_0920 + timedelta(minutes=15)
    engine.expire_pending_evaluations(as_of_time=clock_now, timeout_seconds=120.0)

    # Invariant: Pending evaluation expired and cleaned up
    assert "5m" not in spot_state._pending_breakout_eval
    snap = spot_state.latest_snapshots["5m"]
    assert snap is not None
    assert snap.rvol is None


def test_concurrent_spot_and_futures_candle_processing():
    """Verify Issue P1 Fix: TechnicalAnalysisEngine is thread-safe under concurrent
    spot and futures candle processing without race conditions or deadlocks.
    """
    import concurrent.futures

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    def feed_futures():
        for i in range(30):
            ts = base_time + timedelta(minutes=i)
            engine.on_candle(
                Candle(
                    symbol="NSE:NIFTY26NOVFUT",
                    timeframe="1m",
                    timestamp=ts,
                    open=25000.0 + i,
                    high=25010.0 + i,
                    low=24990.0 + i,
                    close=25005.0 + i,
                    volume=1000 + i * 50,
                    is_closed=True,
                )
            )

    def feed_spot():
        for i in range(30):
            ts = base_time + timedelta(minutes=i)
            engine.on_candle(
                Candle(
                    symbol="NSE:NIFTY50-INDEX",
                    timeframe="1m",
                    timestamp=ts,
                    open=25000.0 + i,
                    high=25010.0 + i,
                    low=24990.0 + i,
                    close=25005.0 + i,
                    volume=0,
                    is_closed=True,
                )
            )

    def query_snapshots():
        for _ in range(30):
            engine.get_snapshot("NSE:NIFTY50-INDEX", timeframe="1m")
            engine.get_snapshot("NSE:NIFTY26NOVFUT", timeframe="1m")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(feed_futures)
        f2 = executor.submit(feed_spot)
        f3 = executor.submit(query_snapshots)
        f1.result()
        f2.result()
        f3.result()

    # Invariant: Thread safety verified; state intact
    snap_spot = engine.get_snapshot("NSE:NIFTY50-INDEX", timeframe="1m")
    snap_fut = engine.get_snapshot("NSE:NIFTY26NOVFUT", timeframe="1m")
    assert snap_spot is not None
    assert snap_fut is not None


def test_previous_session_candles_arriving_late_rejected():
    """Verify Issue P2 Fix: Late candles from an older trading session are strictly rejected
    and do NOT reset current session date, daily reference levels, or active setups.
    """
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    monday_time = IST.localize(datetime(2026, 11, 23, 10, 0))
    friday_time = IST.localize(datetime(2026, 11, 20, 15, 25))

    engine.set_daily_references(
        symbol="NSE:NIFTY50-INDEX",
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # Monday candle initializes Monday session
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="5m",
            timestamp=monday_time,
            open=25050.0,
            high=25070.0,
            low=25040.0,
            close=25060.0,
            volume=0,
            is_closed=True,
        )
    )

    state = engine.states["NSE:NIFTY50-INDEX"]
    assert state.current_session_date == monday_time.date()
    assert state.daily_reference_levels is not None

    # Late Friday candle arrives
    friday_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="5m",
        timestamp=friday_time,
        open=24800.0,
        high=24820.0,
        low=24790.0,
        close=24810.0,
        volume=0,
        is_closed=True,
    )
    engine.on_candle(friday_candle)

    # Invariant: Monday session was NOT reset by late Friday candle!
    assert state.current_session_date == monday_time.date()
    assert state.daily_reference_levels is not None
    assert state.daily_reference_levels["PDH"] == 25100.0




