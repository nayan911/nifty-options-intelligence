"""Tests for sliding-window rate limiting, option chain OI/IV flow, and feed-gap backfill."""

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytz

from nifty_engine.analysis.derivatives.oi_analysis import OpenInterestAnalyzer
from nifty_engine.core import rate_limiter as rl
from nifty_engine.core.interfaces import OptionChainSnapshot, OptionChainStrike
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.fyers.rest_client import classify_rate_limit

IST = pytz.timezone("Asia/Kolkata")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# --------------------------------------------------------------------------- rate limiter
def test_per_second_window_never_exceeds_cap():
    """Old token bucket allowed ~2x rate inside 1s (burst + refill). Sliding window must not."""
    clock = Clock()
    with patch.object(rl.time, "monotonic", clock):
        lim = AsyncTokenBucketRateLimiter(rate_per_sec=4, rate_per_min=100, max_daily_requests=1000)
        assert all(lim.try_acquire() for _ in range(4))
        clock.t += 0.5
        assert lim.try_acquire() is False  # still inside the same 1s window
        clock.t += 0.51
        assert lim.try_acquire() is True


def test_per_minute_window_blocks_until_oldest_expires():
    clock = Clock()
    with patch.object(rl.time, "monotonic", clock):
        lim = AsyncTokenBucketRateLimiter(rate_per_sec=100, rate_per_min=3, max_daily_requests=1000)
        for _ in range(3):
            assert lim.try_acquire()
            clock.t += 2
        assert lim.try_acquire() is False
        clock.t += 55  # oldest request now >60s old
        assert lim.try_acquire() is True


def test_daily_usage_persists_across_restarts_and_resets_next_ist_day(tmp_path):
    f = tmp_path / "usage.json"
    with patch.object(rl, "_ist_today", return_value=date(2026, 11, 23)):
        a = AsyncTokenBucketRateLimiter(rate_per_sec=100, rate_per_min=100, max_daily_requests=10, usage_file=f)
        for _ in range(3):
            a.acquire_sync()
        a.record_rate_limit_hit()

        b = AsyncTokenBucketRateLimiter(rate_per_sec=100, rate_per_min=100, max_daily_requests=10, usage_file=f)
        assert b.daily_requests_count == 3
        assert b.rate_limit_hits_today == 1

    with patch.object(rl, "_ist_today", return_value=date(2026, 11, 25)):
        c = AsyncTokenBucketRateLimiter(rate_per_sec=100, rate_per_min=100, max_daily_requests=10, usage_file=f)
        assert c.daily_requests_count == 0
        assert c.get_status()["rate_limit_hits_today"] == 0


def test_classify_rate_limit():
    assert classify_rate_limit({"s": "ok"}) is None
    assert classify_rate_limit({"s": "error", "message": "Invalid symbol"}) is None
    assert classify_rate_limit({"s": "error", "code": -429, "message": "x requests/second"}) == "second"
    assert classify_rate_limit({"s": "error", "message": "limit of 50 requests/minute"}) == "minute"
    assert classify_rate_limit({"s": "error", "code": -429, "message": "daily limit reached"}) == "day"
    assert classify_rate_limit({"s": "error", "code": -429, "message": "throttled"}) == "unknown"


# --------------------------------------------------------------------------- OI / IV flow
def _chain(ts, spot, ce_oi, pe_oi, ce_iv, pe_iv):
    strikes = []
    for k in (25350.0, 25400.0, 25450.0):
        strikes.append(
            OptionChainStrike(
                strike=k,
                call={"symbol": f"CE{k}", "ltp": 50.0, "oi": ce_oi[k], "volume": 1000, "iv": ce_iv},
                put={"symbol": f"PE{k}", "ltp": 40.0, "oi": pe_oi[k], "volume": 1000, "iv": pe_iv},
            )
        )
    return OptionChainSnapshot(
        underlying="NSE:NIFTY50-INDEX", spot_price=spot, api_timestamp=ts,
        expiry=date(2026, 12, 1), strikes=strikes,
    )


def test_oi_profile_reports_atm_iv_change_and_near_atm_oi_flow():
    an = OpenInterestAnalyzer(near_atm_strikes_count=1)
    t0 = IST.localize(datetime(2026, 11, 23, 13, 0))
    base_ce = {25350.0: 100_000, 25400.0: 200_000, 25450.0: 300_000}
    base_pe = {25350.0: 300_000, 25400.0: 200_000, 25450.0: 100_000}
    p0 = an.on_option_chain(_chain(t0, 25410, base_ce, base_pe, 14.0, 15.0), as_of=t0)
    assert p0.atm_strike == 25400.0 and p0.atm_iv == 14.5 and p0.atm_iv_change is None

    ce1 = {k: v + 10_000 for k, v in base_ce.items()}  # call writing near ATM
    pe1 = {k: v - 5_000 for k, v in base_pe.items()}
    t1 = t0 + timedelta(seconds=60)
    p1 = an.on_option_chain(_chain(t1, 25405, ce1, pe1, 13.0, 14.0), as_of=t1)
    assert p1.atm_iv == 13.5
    assert p1.atm_iv_change == -1.0
    assert p1.near_atm_call_oi_change == 30_000
    assert p1.near_atm_put_oi_change == -15_000
    assert p1.atm_iv_change_strike == 25400.0
    assert p1.observation_interval_seconds == 60.0


def test_atm_iv_change_uses_same_strike_when_atm_shifts():
    """Spot crossing a strike must not report the smile difference between strikes as an IV move."""
    an = OpenInterestAnalyzer(near_atm_strikes_count=1)
    t0 = IST.localize(datetime(2026, 11, 23, 15, 2))
    oi = {25350.0: 1, 25400.0: 1, 25450.0: 1}

    def chain(ts, spot, ivs):
        strikes = [
            OptionChainStrike(
                strike=k,
                call={"ltp": 10.0, "oi": 1, "volume": 1, "iv": ivs[k]},
                put={"ltp": 10.0, "oi": 1, "volume": 1, "iv": ivs[k]},
            )
            for k in oi
        ]
        return OptionChainSnapshot(underlying="NIFTY", spot_price=spot, api_timestamp=ts,
                                   expiry=date(2026, 11, 23), strikes=strikes)

    smile = {25350.0: 34.0, 25400.0: 29.5, 25450.0: 27.0}
    an.on_option_chain(chain(t0, 25360, smile), as_of=t0)
    # Spot moves to 25395 -> ATM label shifts to 25400; 25350's own IV fell only 0.5
    smile2 = {25350.0: 33.5, 25400.0: 29.3, 25450.0: 26.9}
    p = an.on_option_chain(chain(t0 + timedelta(seconds=60), 25395, smile2), as_of=t0 + timedelta(seconds=60))
    assert p.atm_strike == 25400.0 and p.atm_iv == 29.3
    assert p.atm_iv_change == -0.5
    assert p.atm_iv_change_strike == 25350.0  # change is labelled with the strike it was measured at


def test_pcr_uses_fyers_expiry_totals_and_keeps_window_separately():
    """Window-only PCR drifts as ATM moves the strike window; FYERS callOi/putOi are expiry-wide."""
    from nifty_engine.data.fyers.rest_client import FyersRestClient

    payload = {
        "callOi": 300_000_000,
        "putOi": 213_000_000,
        "expiryData": [{"date": "23-11-2026", "expiry": 1795417200}],
        "optionsChain": [
            {"symbol": "NSE:NIFTY50-INDEX", "strike_price": -1, "ltp": 25380.0},
            {"symbol": "NSE:NIFTY26NOV25400CE", "strike_price": 25400, "option_type": "CE", "ltp": 30.0, "oi": 150},
            {"symbol": "NSE:NIFTY26NOV25400PE", "strike_price": 25400, "option_type": "PE", "ltp": 40.0, "oi": 135},
        ],
    }
    snap = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert snap.total_call_oi == 300_000_000 and snap.total_put_oi == 213_000_000
    assert snap.pcr == 0.71
    assert (snap.window_call_oi, snap.window_put_oi) == (150, 135)

    del payload["callOi"], payload["putOi"]
    fallback = FyersRestClient.parse_option_chain_payload(payload, "NSE:NIFTY50-INDEX")
    assert fallback.total_call_oi == 150 and fallback.pcr == 0.9
