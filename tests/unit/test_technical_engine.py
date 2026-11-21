"""Unit tests for TechnicalAnalysisEngine and SymbolTechnicalState."""

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


def test_symbol_technical_state_indicators_and_trend_bias():
    """Verify streaming indicator computation and trend bias classification."""
    state = SymbolTechnicalState(symbol="NSE:NIFTY26NOVFUT", is_index=False)
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Generate 25 bullish trending 1m candles with volume
    snapshots = []
    price = 25000.0
    for i in range(25):
        price += 10.0
        c = Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=price - 5.0,
            high=price + 8.0,
            low=price - 6.0,
            close=price,
            volume=2000 + (i * 100),
            is_closed=True,
        )
        snap = state.process_1m_candle(c)
        snapshots.append(snap)

    last_snap = snapshots[-1]
    assert last_snap.close == 25250.0
    assert last_snap.ema_9 is not None
    assert last_snap.ema_21 is not None
    assert last_snap.rsi is not None
    assert 50.0 < last_snap.rsi <= 100.0  # Upward trending RSI
    assert last_snap.atr is not None
    assert last_snap.atr > 0
    assert last_snap.vwap is not None
    assert last_snap.vwap_upper is not None
    assert last_snap.vwap_lower is not None
    assert last_snap.trend_bias == "BULLISH"
    assert last_snap.rvol is not None


def test_spot_index_futures_rvol_pairing():
    """Verify that spot index receives futures RVOL and keeps VWAP as None."""
    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # 1. Feed Futures candle with high volume surge (RVOL > 1.0)
    fut_candle = Candle(
        symbol="NSE:NIFTY26NOVFUT",
        timeframe="1m",
        timestamp=base_time,
        open=25000.0,
        high=25030.0,
        low=24990.0,
        close=25025.0,
        volume=50000,
        is_closed=True,
    )
    fut_snap = engine.on_candle(fut_candle)
    assert fut_snap.vwap is not None
    assert fut_snap.rvol is not None

    # 2. Feed Spot Index candle with zero volume
    spot_candle = Candle(
        symbol="NSE:NIFTY50-INDEX",
        timeframe="1m",
        timestamp=base_time,
        open=25000.0,
        high=25028.0,
        low=24992.0,
        close=25020.0,
        volume=0,
        is_closed=True,
    )
    spot_snap = engine.on_candle(spot_candle)
    # Spot index must have NO vwap because spot volume is 0
    assert spot_snap.vwap is None
    # Spot index inherits paired futures RVOL
    assert spot_snap.rvol == fut_snap.rvol


def test_5m_bar_synthesis():
    """Verify that five 1m candles correctly synthesize into one closed 5m candle."""
    state = SymbolTechnicalState(symbol="NSE:NIFTY50-INDEX", is_index=True)
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Feed minutes 9:15, 9:16, 9:17, 9:18, 9:19
    closes = [25010.0, 25020.0, 25005.0, 25030.0, 25025.0]
    highs = [25015.0, 25025.0, 25010.0, 25035.0, 25030.0]
    lows = [24995.0, 25000.0, 24990.0, 25005.0, 25010.0]

    for i in range(5):
        c = Candle(
            symbol="NSE:NIFTY50-INDEX",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=i),
            open=25000.0 if i == 0 else closes[i - 1],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=0,
            is_closed=True,
        )
        state.process_1m_candle(c)

    # 5m bar should be synthesized and closed
    assert len(state.candles_5m) == 1
    bar_5m = state.candles_5m[0]
    assert bar_5m.timeframe == "5m"
    assert bar_5m.open == 25000.0
    assert bar_5m.high == 25035.0  # max of highs
    assert bar_5m.low == 24990.0   # min of lows
    assert bar_5m.close == 25025.0 # close of 9:19 candle
    assert bar_5m.is_closed is True


def test_technical_engine_breakout_callback_dispatch():
    """Verify that high-volume breakout and retest triggers the on_setup_eligible callback."""
    triggered_setups = []

    def handle_setup(setup: BreakoutContext, snap: TechnicalSnapshot):
        triggered_setups.append((setup, snap))

    engine = TechnicalAnalysisEngine(
        primary_index="NSE:NIFTY50-INDEX",
        primary_futures="NSE:NIFTY26NOVFUT",
        on_setup_eligible=handle_setup,
    )
    base_time = IST.localize(datetime(2026, 11, 23, 9, 15))

    # Set daily resistance at 25100 (PDH)
    engine.set_daily_references(
        symbol="NSE:NIFTY26NOVFUT",
        pdh=25100.0,
        pdl=24900.0,
        pdc=25000.0,
        cdo=25020.0,
    )

    # 1. Establish baseline bars below resistance (25080)
    for i in range(20):
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

    # 2. Bullish Breakout bar above 25100 with massive volume (3000 vs 1000 avg => RVOL ~ 3.0)
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=20),
            open=25085.0,
            high=25120.0,
            low=25080.0,
            close=25115.0,
            volume=3500,
            is_closed=True,
        )
    )

    # 3. Retest bar: price dips back to test 25100 (low=25098), closes green at 25108
    engine.on_candle(
        Candle(
            symbol="NSE:NIFTY26NOVFUT",
            timeframe="1m",
            timestamp=base_time + timedelta(minutes=21),
            open=25102.0,
            high=25112.0,
            low=25098.0,
            close=25108.0,
            volume=1200,
            is_closed=True,
        )
    )

    # The setup should transition to SETUP_ELIGIBLE and dispatch callback!
    assert len(triggered_setups) >= 1
    setup, snap = triggered_setups[0]
    assert setup.current_state == BreakoutState.SETUP_ELIGIBLE
    assert setup.direction == BreakoutDirection.BULLISH
    assert setup.stop_loss is not None
    assert setup.target_price is not None
    assert setup.stop_loss < 25108.0 < setup.target_price
