"""Unit tests for FyersRestClient option chain parsing."""


import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from nifty_engine.data.fyers.rest_client import FyersRestClient
from nifty_engine.utils.timezone import now_ist


def test_parse_option_chain_flat_contract_records():
    """Verify parsing of FYERS official flat optionsChain structure with individual CE and PE rows."""
    payload = {
        "underlyingValue": 24125.50,
        "optionsChain": [
            # 24100 CE
            {
                "symbol": "NSE:NIFTY26NOV24100CE",
                "strike_price": 24100.0,
                "option_type": "CE",
                "ltp": 145.2,
                "oi": 120000,
                "oichange": 5000,
                "volume": 350000,
                "bid": 145.0,
                "ask": 145.4,
                "expiry": 1795687200,
                "greeks": {
                    "iv": 13.8,
                    "delta": 0.55,
                    "theta": -12.5,
                    "gamma": 0.0025,
                    "vega": 14.8,
                },
            },
            # 24100 PE
            {
                "symbol": "NSE:NIFTY26NOV24100PE",
                "strike_price": 24100.0,
                "option_type": "PE",
                "ltp": 85.6,
                "oi": 180000,
                "oichange": 12000,
                "volume": 420000,
                "bid": 85.5,
                "ask": 85.7,
                "expiry": 1795687200,
                "greeks": {
                    "iv": 14.2,
                    "delta": -0.45,
                    "theta": -10.2,
                    "gamma": 0.0024,
                    "vega": 14.6,
                },
            },
            # 24150 CE
            {
                "symbol": "NSE:NIFTY26NOV24150CE",
                "strike_price": 24150.0,
                "option_type": "CE",
                "ltp": 115.0,
                "oi": 95000,
                "oichange": 3000,
                "volume": 280000,
                "bid": 114.8,
                "ask": 115.2,
                "expiry": 1795687200,
                "greeks": {"iv": 13.5, "delta": 0.48, "theta": -11.8},
            },
        ],
    }

    snapshot = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")

    assert snapshot.underlying == "NIFTY"
    assert snapshot.spot_price == 24125.50
    assert len(snapshot.strikes) == 2  # 24100 and 24150

    # Verify 24100 strike
    s_24100 = next(s for s in snapshot.strikes if s.strike == 24100.0)
    assert s_24100.call is not None
    assert s_24100.put is not None
    assert s_24100.call["symbol"] == "NSE:NIFTY26NOV24100CE"
    assert s_24100.call["ltp"] == 145.2
    assert s_24100.call["oi"] == 120000
    assert s_24100.call["delta"] == 0.55
    assert s_24100.put["symbol"] == "NSE:NIFTY26NOV24100PE"
    assert s_24100.put["oi"] == 180000

    # Verify 24150 strike
    s_24150 = next(s for s in snapshot.strikes if s.strike == 24150.0)
    assert s_24150.call is not None
    assert s_24150.put is None

    # Verify totals and PCR
    assert snapshot.total_call_oi == 120000 + 95000
    assert snapshot.total_put_oi == 180000
    expected_pcr = round(180000 / (120000 + 95000), 3)
    assert snapshot.pcr == expected_pcr


def test_parse_option_chain_nested_records():
    """Verify parsing of nested 'ce' / 'pe' format with expiryData."""
    payload = {
        "underlyingValue": 24000.0,
        "expiryData": [{"date": "23-Nov-2026", "timestamp": 1795392000}],
        "optionsChain": [
            {
                "strike_price": 24000.0,
                "ce": {"symbol": "CE24000", "ltp": 100.0, "oi": 50000, "volume": 1000},
                "pe": {"symbol": "PE24000", "ltp": 90.0, "oi": 60000, "volume": 1200},
            }
        ],
    }
    snapshot = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert len(snapshot.strikes) == 1
    s = snapshot.strikes[0]
    assert s.strike == 24000.0
    assert s.call["ltp"] == 100.0
    assert s.put["ltp"] == 90.0
    assert snapshot.total_call_oi == 50000
    assert snapshot.total_put_oi == 60000
    assert snapshot.pcr == 1.2
    assert snapshot.expiry.year == 2026


def test_parse_option_chain_official_fyers_oich_and_greeks():
    """Verify extraction of official FYERS 'oich' key and Greeks."""
    payload = {
        "underlyingValue": 25780.0,
        "expiryData": [{"date": "23-Nov-2026", "timestamp": 1795392000}],
        "optionsChain": [
            {
                "symbol": "NSE:NIFTY26NOV25800CE",
                "strike_price": 25800.0,
                "option_type": "CE",
                "ltp": 110.5,
                "oi": 80000,
                "oich": 4500,  # Official FYERS field name
                "volume": 200000,
                "bid": 110.2,
                "ask": 110.8,
                "greeks": {
                    "iv": 14.2,
                    "delta": 0.52,
                    "gamma": 0.003,
                    "theta": -12.1,
                    "vega": 15.4,
                },
            }
        ],
    }
    snapshot = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert len(snapshot.strikes) == 1
    c = snapshot.strikes[0].call
    assert c is not None
    assert c["oi_change"] == 4500
    assert c["gamma"] == 0.003
    assert c["vega"] == 15.4
    assert c["bid"] == 110.2
    assert c["ask"] == 110.8


def test_underlying_identification_banknifty_vs_nifty():
    """Verify BANKNIFTY is not accidentally misclassified as NIFTY."""
    payload = {
        "underlyingValue": 52000.0,
        "expiryData": [{"date": "25-Nov-2026", "timestamp": 1795564800}],
        "optionsChain": [
            {
                "symbol": "NSE:BANKNIFTY26NOV52000CE",
                "strike_price": 52000.0,
                "option_type": "CE",
                "ltp": 350.0,
                "oi": 25000,
            }
        ],
    }

    snap_bank = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTYBANK-INDEX")
    assert snap_bank.underlying == "BANKNIFTY"

    snap_nifty = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert snap_nifty.underlying == "NIFTY"


def test_missing_expiry_raises_error():
    """Verify that an option chain payload with no detectable expiry raises ValueError."""
    payload = {
        "underlyingValue": 25000.0,
        "optionsChain": [
            {
                "symbol": "NSE:NIFTY26NOV25000CE",
                "strike_price": 25000.0,
                "option_type": "CE",
                "ltp": 50.0,
                "oi": 1000,
            }
        ],
    }
    with pytest.raises(ValueError, match="Could not resolve contract expiry date"):
        FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")


def test_fyers_option_chain_parser_synchronization_and_timestamp_provenance():
    """Verify that production FYERS snapshots default to is_synchronized_snapshot=False and extract explicit timestamps."""
    from datetime import datetime

    import pytz

    from nifty_engine.analysis.derivatives.contract_selector import OptionContractSelector
    from nifty_engine.analysis.derivatives.oi_analysis import OpenInterestAnalyzer
    from nifty_engine.analysis.technical.breakouts import BreakoutDirection

    ist = pytz.timezone("Asia/Kolkata")
    t0 = ist.localize(datetime(2026, 11, 23, 10, 0))

    # 1. Standard production FYERS payload (no synchronicity guarantee)
    payload_prod = {
        "underlyingValue": 25100.0,
        "expiryData": [{"date": "26-Nov-2026", "timestamp": 1795687200}],
        "optionsChain": [
            {
                "symbol": "NSE:NIFTY26NOV25100CE",
                "strike_price": 25100.0,
                "option_type": "CE",
                "ltp": 95.0,
                "oi": 140000,
                "volume": 45000,
                "bid": 94.5,
                "ask": 95.5,
                "greeks": {
                    "iv": 14.5,
                    "delta": 0.50,
                    "gamma": 0.0004,
                    "theta": -8.5,
                    "vega": 12.0,
                },
            }
        ],
    }

    snap_prod = FyersRestClient.parse_option_chain_payload(payload_prod, "NSE:NIFTY50-INDEX")
    # Must NOT assume synchronization by default
    assert snap_prod.is_synchronized_snapshot is False

    t0 = snap_prod.api_timestamp

    oi_analyzer = OpenInterestAnalyzer()
    profile = oi_analyzer.on_option_chain(snap_prod)
    strike_info = profile.strikes[25100.0]
    # Provenance must be INFERRED and timestamps must be None
    assert strike_info.call_greeks_provenance == "INFERRED"
    assert strike_info.call_greeks_timestamp is None
    assert strike_info.call_quote_timestamp is None

    # Downstream contract selector without quote_provider must REJECT because quote timestamp is missing
    selector_unverified = OptionContractSelector()
    res_rejected = selector_unverified.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res_rejected is None

    # When supplied with a verifiably fresh on-demand quote provider, it safely enters and calculates Black-76 Greeks
    def fresh_q_provider(sym):
        return {
            "bid": 94.5,
            "ask": 95.5,
            "ltp": 95.0,
            "timestamp": t0,
        }

    selector_verified = OptionContractSelector(quote_provider=fresh_q_provider)
    res = selector_verified.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=t0,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res is not None
    assert res.greeks.model_used == "BLACK_76"
    assert res.greeks.is_calculated is True

    # 2. Payload with explicit broker greeks timestamp
    payload_with_ts = {
        "underlyingValue": 25100.0,
        "expiryData": [{"date": "26-Nov-2026", "timestamp": 1795687200}],
        "optionsChain": [
            {
                "symbol": "NSE:NIFTY26NOV25100CE",
                "strike_price": 25100.0,
                "option_type": "CE",
                "ltp": 95.0,
                "oi": 140000,
                "volume": 45000,
                "bid": 94.5,
                "ask": 95.5,
                "quote_timestamp": int(t0.timestamp()),
                "greeks": {
                    "iv": 14.5,
                    "delta": 0.50,
                    "gamma": 0.0004,
                    "theta": -8.5,
                    "vega": 12.0,
                    "greeks_timestamp": int(t0.timestamp()),
                },
            }
        ],
    }
    snap_explicit = FyersRestClient.parse_option_chain_payload(payload_with_ts, "NSE:NIFTY50-INDEX")
    profile_explicit = oi_analyzer.on_option_chain(snap_explicit)
    assert profile_explicit.strikes[25100.0].call_greeks_provenance == "EXPLICIT"
    assert profile_explicit.strikes[25100.0].call_greeks_timestamp is not None
    assert profile_explicit.strikes[25100.0].call_quote_timestamp is not None


def test_timestampless_option_chain_rejected_unless_quote_refreshed():
    """Regression test: Newly received option chain with absent bid/ask timestamps is rejected

    unless an independent, verifiably fresh quote is supplied.
    """
    from datetime import datetime, timedelta

    import pytz

    from nifty_engine.analysis.derivatives.contract_selector import OptionContractSelector
    from nifty_engine.analysis.derivatives.oi_analysis import OpenInterestAnalyzer
    from nifty_engine.analysis.technical.breakouts import BreakoutDirection

    ist = pytz.timezone("Asia/Kolkata")
    now = ist.localize(datetime(2026, 11, 23, 10, 30))

    # Newly received option-chain response, but bid/ask timestamps are absent
    payload = {
        "underlyingValue": 25100.0,
        "expiryData": [{"date": "26-Nov-2026", "timestamp": 1795687200}],
        "optionsChain": [
            {
                "symbol": "NSE:NIFTY26NOV25100CE",
                "strike_price": 25100.0,
                "option_type": "CE",
                "ltp": 95.0,
                "oi": 150000,
                "volume": 50000,
                "bid": 94.5,
                "ask": 95.5,
                # No quote_timestamp or tt present!
            }
        ],
    }

    snap = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert snap.is_synchronized_snapshot is False

    analyzer = OpenInterestAnalyzer()
    profile = analyzer.on_option_chain(snap)

    strike = profile.strikes[25100.0]
    # Timestamp must not be replaced with API receipt time
    assert strike.call_quote_timestamp is None

    # Case 1: Selector without quote_provider must reject due to missing timestamp
    selector_default = OptionContractSelector()
    res1 = selector_default.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=now,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res1 is None, "Expected contract rejection when quote timestamp is absent"

    # Case 2: Selector with stale quote_provider must reject due to stale quote age
    def stale_provider(sym):
        return {
            "bid": 94.5,
            "ask": 95.5,
            "ltp": 95.0,
            "timestamp": now - timedelta(seconds=60),
        }

    selector_stale = OptionContractSelector(quote_provider=stale_provider)
    res2 = selector_stale.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=now,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res2 is None, "Expected contract rejection when refreshed quote is stale"

    # Case 3: Selector with fresh verified quote_provider must qualify
    def fresh_provider(sym):
        return {
            "bid": 94.5,
            "ask": 95.5,
            "ltp": 95.0,
            "timestamp": now,
        }

    selector_fresh = OptionContractSelector(quote_provider=fresh_provider)
    res3 = selector_fresh.select_contract(
        direction=BreakoutDirection.BULLISH,
        spot_price=25100.0,
        chain_profile=profile,
        as_of=now,
        risk_points=15.0,
        target_points_t1=45.0,
    )
    assert res3 is not None, "Expected contract selection when verifiably fresh quote is provided"
    assert res3.symbol == "NSE:NIFTY26NOV25100CE"
    assert res3.quote_timestamp == now


def test_get_historical_candles_candle_completeness():
    """Verify that forming candles have is_closed=False while past completed candles have is_closed=True."""
    mock_auth = MagicMock()
    mock_client = MagicMock()
    mock_auth.get_client.return_value = mock_client

    rest_client = FyersRestClient(auth_provider=mock_auth)

    now = now_ist()
    # Candle 1: started 15 minutes ago (5m bar ended 10 minutes ago -> closed)
    ts_past = int((now - timedelta(minutes=15)).timestamp())
    # Candle 2: started 1 minute ago (5m bar ends in 4 minutes -> still forming)
    ts_forming = int((now - timedelta(minutes=1)).timestamp())

    mock_client.history.return_value = {
        "s": "ok",
        "candles": [
            [ts_past, 25000.0, 25050.0, 24980.0, 25020.0, 5000],
            [ts_forming, 25020.0, 25030.0, 25010.0, 25025.0, 1000],
        ],
    }

    # 1. Sync fetch
    candles_sync = rest_client.get_historical_candles_sync("NSE:NIFTY50-INDEX", resolution="5")
    assert len(candles_sync) == 2
    assert candles_sync[0].is_closed is True
    assert candles_sync[1].is_closed is False

    # 2. Async fetch
    async def _test_async():
        return await rest_client.get_historical_candles("NSE:NIFTY50-INDEX", resolution="5")

    candles_async = asyncio.run(_test_async())
    assert len(candles_async) == 2
    assert candles_async[0].is_closed is True
    assert candles_async[1].is_closed is False


def test_get_historical_candles_sync_rate_limit_backoff(monkeypatch):
    """Broker limit responses: per-second waits ~1.1s; per-minute waits out the full 61s window,
    and a repeated per-minute violation aborts instead of risking FYERS' >3/day account block."""
    import pytest

    from nifty_engine.data.fyers.rest_client import FyersRateLimitError

    mock_auth = MagicMock()
    mock_client = MagicMock()
    mock_auth.get_client.return_value = mock_client
    mock_limiter = MagicMock()
    hits = {"n": 0}

    def _hit():
        hits["n"] += 1
        return hits["n"]

    mock_limiter.record_rate_limit_hit.side_effect = _hit

    rest_client = FyersRestClient(auth_provider=mock_auth, rate_limiter=mock_limiter)

    sleep_calls = []
    monkeypatch.setattr("nifty_engine.data.fyers.rest_client.time_mod.sleep", lambda s: sleep_calls.append(s))

    now = now_ist()
    ts = int((now - timedelta(minutes=15)).timestamp())
    success_resp = {
        "s": "ok",
        "candles": [[ts, 25000.0, 25050.0, 24980.0, 25020.0, 5000]],
    }

    # 1. Per-second throttling: short fixed wait, retried
    mock_client.history.side_effect = [
        {"s": "error", "code": -429, "message": "Reached limit of 5 requests/second"},
        {"s": "error", "message": "Too many requests / second limit"},
        success_resp,
    ]
    candles = rest_client.get_historical_candles_sync("NSE:NIFTY50-INDEX", resolution="5")
    assert len(candles) == 1
    assert sleep_calls == [1.1, 1.1]

    # 2. First per-minute violation of the day: wait out the full rolling minute, then retry
    sleep_calls.clear()
    hits["n"] = 0
    mock_client.history.side_effect = [
        {"s": "error", "code": -429, "message": "Standard plan limit of 50 requests/minute"},
        success_resp,
    ]
    candles_min = rest_client.get_historical_candles_sync("NSE:NIFTY50-INDEX", resolution="5")
    assert len(candles_min) == 1
    assert sleep_calls == [61.0]

    # 3. Second per-minute violation today: refuse to retry
    sleep_calls.clear()
    mock_client.history.side_effect = [
        {"s": "error", "code": -429, "message": "requests/minute limit exceeded"},
        success_resp,
    ]
    with pytest.raises(FyersRateLimitError):
        rest_client.get_historical_candles_sync("NSE:NIFTY50-INDEX", resolution="5")
    assert sleep_calls == []

    # 4. Non-limit errors are not mistaken for rate limits
    mock_client.history.side_effect = [{"s": "error", "message": "Invalid symbol"}]
    with pytest.raises(RuntimeError, match="Invalid symbol"):
        rest_client.get_historical_candles_sync("NSE:BAD", resolution="5")
