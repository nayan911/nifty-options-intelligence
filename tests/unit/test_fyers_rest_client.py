"""Unit tests for FyersRestClient option chain parsing."""


from nifty_engine.data.fyers.rest_client import FyersRestClient


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
    import pytest
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
