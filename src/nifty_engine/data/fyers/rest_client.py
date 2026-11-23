"""Rate-limited, asynchronous REST Client for FYERS Data API v3."""

import asyncio
import logging
import time as time_mod
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz
from fyers_apiv3 import fyersModel

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

from nifty_engine.core.interfaces import (
    Candle,
    OptionChainSnapshot,
    OptionChainStrike,
)
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.fyers.auth import FyersAuthProvider
from nifty_engine.utils.timezone import now_ist


class FyersRateLimitError(RuntimeError):
    """Broker rejected a request for exceeding a rate limit (FYERS code -429)."""

    def __init__(self, message: str, scope: str):
        super().__init__(message)
        self.scope = scope  # "second" | "minute" | "day" | "unknown"


def classify_rate_limit(response: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the violated window ("second"/"minute"/"day"/"unknown") or None if not a limit error."""
    if not response or response.get("s") == "ok":
        return None
    code = response.get("code")
    msg = str(response.get("message", "")).lower()
    is_limit = code in (-429, 429) or "limit" in msg or "too many" in msg
    if not is_limit:
        return None
    if "day" in msg or "daily" in msg:
        return "day"
    if "minute" in msg:
        return "minute"
    if "second" in msg:
        return "second"
    return "unknown"


# Seconds to wait before retrying after a broker-side limit response. A per-minute
# violation must wait out the full rolling minute: retrying sooner risks a second
# violation, and FYERS blocks the account for the day after more than 3.
RATE_LIMIT_BACKOFF = {"second": 1.1, "minute": 61.0, "unknown": 61.0}


class FyersRestClient:
    """Provides safe, rate-limited access to FYERS Data endpoints."""

    def __init__(
        self,
        auth_provider: Optional[FyersAuthProvider] = None,
        rate_limiter: Optional[AsyncTokenBucketRateLimiter] = None,
    ) -> None:
        self.auth_provider = auth_provider or FyersAuthProvider()
        self.rate_limiter = rate_limiter or AsyncTokenBucketRateLimiter()
        self._fyers_client: Optional[fyersModel.FyersModel] = None

    def _limit_backoff(
        self,
        response: Optional[Dict[str, Any]],
        symbol: str,
        attempt: int,
        max_retries: int,
    ) -> Optional[float]:
        """If response is a broker rate-limit rejection, record it and return the retry delay.

        Returns None when the response is not a limit error. Raises FyersRateLimitError
        when retrying would be unsafe (daily cap, or repeated per-minute violations).
        """
        scope = classify_rate_limit(response)
        if scope is None:
            return None
        hits = self.rate_limiter.record_rate_limit_hit()
        msg = str((response or {}).get("message", "rate limited"))
        if scope == "day":
            raise FyersRateLimitError(f"FYERS daily limit reached ({symbol}): {msg}", scope)
        if scope in ("minute", "unknown") and hits >= 2:
            # One more per-minute violation edges toward the >3/day account block.
            raise FyersRateLimitError(
                f"FYERS per-minute limit hit {hits}x today ({symbol}); refusing to retry: {msg}", scope
            )
        delay = RATE_LIMIT_BACKOFF.get(scope, 61.0)
        logger.warning(
            "FYERS %s rate limit for %s (attempt %d/%d). Sleeping %.1fs...",
            scope, symbol, attempt + 1, max_retries, delay,
        )
        return delay

    def get_quote(self, symbol: str, blocking: bool = False) -> Optional[Dict[str, Any]]:
        """Rate-limited single-symbol quote. Non-blocking by default: returns None if no budget.

        Quotes are opportunistic (paper fills, contract checks) and run on latency-sensitive
        paths, so they must never stall behind the per-minute window or bypass the limiter.
        """
        if blocking:
            self.rate_limiter.acquire_sync()
        elif not self.rate_limiter.try_acquire():
            logger.debug("Quote for %s skipped: no API budget in current window", symbol)
            return None
        response = self._get_client().quotes(data={"symbols": symbol})
        if classify_rate_limit(response):
            self.rate_limiter.record_rate_limit_hit()
            return None
        return response

    def _get_client(self) -> fyersModel.FyersModel:
        if self._fyers_client is None:
            self._fyers_client = self.auth_provider.get_client()
        return self._fyers_client

    async def get_historical_candles(
        self,
        symbol: str,
        resolution: str = "1",
        date_from: str = "",
        date_to: str = "",
        max_retries: int = 3,
    ) -> List[Candle]:
        """Fetch historical candles with rate limiting and exponential backoff on transient throttling.

        resolution: '1', '5', '15', 'D'
        """
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": date_from,
            "range_to": date_to,
            "cont_flag": "1",
        }

        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = None
        error_msg = ""

        for attempt in range(max_retries):
            await self.rate_limiter.acquire()
            response = await loop.run_in_executor(None, lambda: client.history(data=data))

            if response and response.get("s") == "ok":
                break

            error_msg = response.get("message", "Error fetching historical data") if response else "Empty response"
            delay = self._limit_backoff(response, symbol, attempt, max_retries)
            if delay is not None:
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(f"FYERS Historical Data Error ({symbol}): {error_msg}")
        else:
            raise RuntimeError(f"FYERS Historical Data Error ({symbol}): {error_msg}")

        raw_candles = response.get("candles", [])
        candles = []
        now = now_ist()
        for c in raw_candles:
            # c = [epoch_timestamp, open, high, low, close, volume]
            ts = datetime.fromtimestamp(c[0], tz=IST)
            if resolution in ("D", "1D", "day"):
                close_dt = IST.localize(datetime.combine(ts.date(), time(15, 30)))
            elif resolution.isdigit():
                close_dt = ts + timedelta(minutes=int(resolution))
            else:
                close_dt = ts + timedelta(minutes=1)
            is_closed = close_dt <= now

            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=f"{resolution}m" if resolution.isdigit() else resolution,
                    timestamp=ts,
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=int(c[5]),
                    is_closed=is_closed,
                )
            )
        return candles

    def get_historical_candles_sync(
        self,
        symbol: str,
        resolution: str = "1",
        date_from: str = "",
        date_to: str = "",
        max_retries: int = 3,
        continuous: bool = True,
        include_oi: bool = False,
    ) -> List[Candle]:
        """Synchronously fetch historical candles via FYERS SDK with rate limiting and backoff.

        continuous=True asks FYERS to stitch futures across expiries (cont_flag=1); the stitched
        series switches contracts at expiry, so OI and price jump at those boundaries.
        include_oi=True requests open interest (oi_flag=1); FYERS then returns a 7th column
        for derivatives (indices/equities still return 6).
        """
        response = None
        error_msg = ""
        for attempt in range(max_retries):
            self.rate_limiter.acquire_sync()

            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": date_from,
                "range_to": date_to,
                "cont_flag": "1" if continuous else "0",
            }
            if include_oi:
                data["oi_flag"] = "1"

            client = self._get_client()
            response = client.history(data=data)

            if response and response.get("s") == "ok":
                break

            error_msg = response.get("message", "Error fetching historical data") if response else "Empty response"
            delay = self._limit_backoff(response, symbol, attempt, max_retries)
            if delay is not None:
                time_mod.sleep(delay)
                continue
            raise RuntimeError(f"FYERS Historical Data Error ({symbol}): {error_msg}")
        else:
            raise RuntimeError(f"FYERS Historical Data Error ({symbol}): {error_msg}")

        raw_candles = response.get("candles", [])
        candles = []
        now = now_ist()
        for c in raw_candles:
            ts = datetime.fromtimestamp(c[0], tz=IST)
            if resolution in ("D", "1D", "day"):
                close_dt = IST.localize(datetime.combine(ts.date(), time(15, 30)))
            elif resolution.isdigit():
                close_dt = ts + timedelta(minutes=int(resolution))
            else:
                close_dt = ts + timedelta(minutes=1)
            is_closed = close_dt <= now

            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=f"{resolution}m" if resolution.isdigit() else resolution,
                    timestamp=ts,
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=int(c[5]),
                    oi=int(c[6]) if (include_oi and len(c) > 6 and c[6] is not None) else None,
                    is_closed=is_closed,
                )
            )
        return candles

    async def get_option_chain(
        self,
        symbol: str = "NSE:NIFTY50-INDEX",
        strikecount: int = 10,
        expiry_epoch: Optional[int] = None,
    ) -> OptionChainSnapshot:
        """Fetch live option chain with Greeks (nearest expiry unless `expiry_epoch` is given)."""
        await self.rate_limiter.acquire()

        data = {
            "symbol": symbol,
            "strikecount": strikecount,
            "timestamp": str(expiry_epoch) if expiry_epoch else "",
            "greeks": "1",
        }

        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.optionchain(data=data))

        scope = classify_rate_limit(response)
        if scope is not None:
            self.rate_limiter.record_rate_limit_hit()
            raise FyersRateLimitError(
                f"FYERS Option Chain rate limited ({scope}): {response.get('message', '')}", scope
            )
        if not response or response.get("s") != "ok":
            error_msg = response.get("message", "Error fetching option chain") if response else "Empty response"
            raise RuntimeError(f"FYERS Option Chain Error ({symbol}): {error_msg}")

        chain_data = response.get("data", {})
        return self.parse_option_chain_payload(chain_data, underlying_symbol=symbol, requested_expiry_epoch=expiry_epoch)

    @staticmethod
    def parse_option_chain_payload(
        chain_data: Dict[str, Any],
        underlying_symbol: str,
        requested_expiry_epoch: Optional[int] = None,
    ) -> OptionChainSnapshot:
        """Parses FYERS optionchain API response payload into normalized OptionChainSnapshot.

        Supports both:
        1. Official FYERS flat array where each item in optionsChain is an individual CE/PE contract.
        2. Nested structure with 'ce' and 'pe' sub-dictionaries per strike.
        """
        options_chain = chain_data.get("optionsChain", [])
        spot_price = float(chain_data.get("underlyingValue", 0.0))

        # Map: strike_price -> {'call': dict, 'put': dict}
        strike_map: Dict[float, Dict[str, Optional[Dict[str, Any]]]] = {}
        total_call_oi = 0
        total_put_oi = 0
        detected_expiry: Optional[date] = None

        # When a specific expiry was requested, FYERS returns that expiry's contracts but still
        # lists ALL expiries in expiryData with the nearest first (verified live: a 01-Dec request
        # returned NIFTY26D01... contracts with expiryData[0] = 23-Nov). Contracts carry no expiry
        # field, so the requested expiry is the only reliable label.
        if requested_expiry_epoch:
            try:
                detected_expiry = datetime.fromtimestamp(int(requested_expiry_epoch), tz=IST).date()
            except (TypeError, ValueError, OSError):
                detected_expiry = None

        # Check top-level expiryData (official FYERS v3 specification)
        expiry_data = chain_data.get("expiryData", [])
        if detected_expiry is None and expiry_data and isinstance(expiry_data, list):
            first_exp = expiry_data[0]
            if isinstance(first_exp, dict):
                raw_exp_ts = first_exp.get("expiry") or first_exp.get("timestamp")
                if raw_exp_ts is not None:
                    try:
                        detected_expiry = datetime.fromtimestamp(int(raw_exp_ts), tz=IST).date()
                    except (ValueError, TypeError):
                        pass
                if detected_expiry is None and first_exp.get("date"):
                    raw_date_str = str(first_exp["date"]).strip()
                    for fmt in ("%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"):
                        try:
                            detected_expiry = datetime.strptime(raw_date_str, fmt).date()
                            break
                        except ValueError:
                            pass
            elif isinstance(first_exp, (int, float, str)):
                try:
                    detected_expiry = datetime.fromtimestamp(int(first_exp), tz=IST).date()
                except (ValueError, TypeError):
                    pass

        for item in options_chain:
            # Check for underlying spot record (FYERS reports index LTP with strike_price = -1)
            raw_sp = float(item.get("strike_price", item.get("strike", 0.0)))
            if raw_sp <= 0:
                if spot_price <= 0.0 and item.get("ltp") is not None:
                    spot_price = float(item["ltp"])
                continue
            # Case 1: Nested structure (item contains 'ce' and/or 'pe')
            if "ce" in item or "pe" in item:
                strike_p = float(item.get("strike_price", 0.0))
                if strike_p not in strike_map:
                    strike_map[strike_p] = {"call": None, "put": None}

                if item.get("ce"):
                    c = item["ce"]
                    oi = int(c.get("oi") or 0)
                    total_call_oi += oi
                    strike_map[strike_p]["call"] = FyersRestClient._extract_contract_fields(c)
                if item.get("pe"):
                    p = item["pe"]
                    oi = int(p.get("oi") or 0)
                    total_put_oi += oi
                    strike_map[strike_p]["put"] = FyersRestClient._extract_contract_fields(p)

                if detected_expiry is None:
                    exp_val = item.get("expiry") or (item.get("ce") or {}).get("expiry")
                    if isinstance(exp_val, (int, float)) and exp_val > 0:
                        detected_expiry = datetime.fromtimestamp(exp_val, tz=IST).date()

            # Case 2: Flat structure (item is a single contract record with option_type / opt)
            else:
                strike_p = float(item.get("strike_price", item.get("strike", 0.0)))
                if strike_p <= 0:
                    continue
                if strike_p not in strike_map:
                    strike_map[strike_p] = {"call": None, "put": None}

                opt_type = str(item.get("option_type", item.get("opt", ""))).upper()
                sym = item.get("symbol", "")
                if not opt_type:
                    if sym.endswith("CE"):
                        opt_type = "CE"
                    elif sym.endswith("PE"):
                        opt_type = "PE"

                contract_dict = FyersRestClient._extract_contract_fields(item)
                oi = int(item.get("oi") or 0)

                if opt_type in ("CE", "CALL"):
                    total_call_oi += oi
                    strike_map[strike_p]["call"] = contract_dict
                elif opt_type in ("PE", "PUT"):
                    total_put_oi += oi
                    strike_map[strike_p]["put"] = contract_dict

                # Extract expiry date if present
                if detected_expiry is None and item.get("expiry"):
                    exp_val = item["expiry"]
                    if isinstance(exp_val, (int, float)) and exp_val > 0:
                        detected_expiry = datetime.fromtimestamp(exp_val, tz=IST).date()

        if detected_expiry is None:
            raise ValueError(
                f"Could not resolve contract expiry date from FYERS option chain payload for {underlying_symbol}."
            )

        parsed_strikes: List[OptionChainStrike] = [
            OptionChainStrike(
                strike=s,
                call=data["call"],
                put=data["put"],
            )
            for s, data in sorted(strike_map.items())
        ]

        window_call_oi, window_put_oi = total_call_oi, total_put_oi

        available_expiries: List[Tuple[date, int]] = []
        for e in expiry_data if isinstance(expiry_data, list) else []:
            if not isinstance(e, dict):
                continue
            try:
                epoch = int(e.get("expiry") or e.get("timestamp"))
                available_expiries.append((datetime.fromtimestamp(epoch, tz=IST).date(), epoch))
            except (TypeError, ValueError):
                continue
        # Prefer FYERS' expiry-wide totals (verified per-expiry: they change with the
        # requested expiry) so PCR doesn't drift as ATM shifts the strike window.
        try:
            exp_call_oi = int(chain_data.get("callOi") or 0)
            exp_put_oi = int(chain_data.get("putOi") or 0)
        except (TypeError, ValueError):
            exp_call_oi = exp_put_oi = 0
        if exp_call_oi > 0 and exp_put_oi > 0:
            total_call_oi, total_put_oi = exp_call_oi, exp_put_oi
        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0.0

        sym_upper = underlying_symbol.upper()
        if "BANKNIFTY" in sym_upper or "NIFTYBANK" in sym_upper:
            underlying = "BANKNIFTY"
        elif "NIFTY" in sym_upper:
            underlying = "NIFTY"
        elif "SENSEX" in sym_upper:
            underlying = "SENSEX"
        else:
            underlying = underlying_symbol

        is_synced = bool(
            chain_data.get("is_synchronized")
            or chain_data.get("is_synchronized_snapshot")
            or chain_data.get("greeks_synchronized", False)
        )

        return OptionChainSnapshot(
            underlying=underlying,
            spot_price=spot_price,
            api_timestamp=datetime.now(IST),
            expiry=detected_expiry,
            strikes=parsed_strikes,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr=pcr,
            window_call_oi=window_call_oi,
            window_put_oi=window_put_oi,
            available_expiries=available_expiries,
            is_synchronized_snapshot=is_synced,
        )

    @staticmethod
    def _extract_contract_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts standard contract fields preserving None for missing/unreported broker fields."""
        greeks = raw.get("greeks", {}) or {}

        # Parse documented FYERS oich / oichange / oi_change
        raw_oich = raw.get("oich")
        if raw_oich is None:
            raw_oich = raw.get("oichange")
        if raw_oich is None:
            raw_oich = raw.get("oi_change")

        # Parse volume
        raw_vol = raw.get("volume")
        if raw_vol is None:
            raw_vol = raw.get("vol_traded_today")

        raw_iv = greeks.get("iv") if greeks.get("iv") is not None else raw.get("iv")
        raw_delta = greeks.get("delta") if greeks.get("delta") is not None else raw.get("delta")
        raw_gamma = greeks.get("gamma") if greeks.get("gamma") is not None else raw.get("gamma")
        raw_theta = greeks.get("theta") if greeks.get("theta") is not None else raw.get("theta")
        raw_vega = greeks.get("vega") if greeks.get("vega") is not None else raw.get("vega")

        # Extract explicit quote and greeks timestamps if present in raw broker record
        quote_ts = None
        raw_q_ts = raw.get("quote_timestamp") or raw.get("timestamp") or raw.get("tt")
        if raw_q_ts is not None:
            if isinstance(raw_q_ts, datetime):
                quote_ts = raw_q_ts
            elif isinstance(raw_q_ts, (int, float)) and raw_q_ts > 0:
                try:
                    quote_ts = datetime.fromtimestamp(int(raw_q_ts), tz=IST)
                except Exception:
                    quote_ts = None

        greeks_ts = None
        raw_g_ts = (
            greeks.get("timestamp")
            or greeks.get("greeks_timestamp")
            or greeks.get("greeks_time")
            or raw.get("greeks_timestamp")
        )
        if raw_g_ts is not None:
            if isinstance(raw_g_ts, datetime):
                greeks_ts = raw_g_ts
            elif isinstance(raw_g_ts, (int, float)) and raw_g_ts > 0:
                try:
                    greeks_ts = datetime.fromtimestamp(int(raw_g_ts), tz=IST)
                except Exception:
                    greeks_ts = None

        return {
            "symbol": raw.get("symbol", ""),
            "ltp": float(raw["ltp"]) if raw.get("ltp") is not None else None,
            "oi": int(raw["oi"]) if raw.get("oi") is not None else None,
            "oi_change": int(raw_oich) if raw_oich is not None else None,
            "volume": int(raw_vol) if raw_vol is not None else None,
            "bid": float(raw["bid"]) if raw.get("bid") is not None else None,
            "ask": float(raw["ask"]) if raw.get("ask") is not None else None,
            "iv": float(raw_iv) if raw_iv is not None else None,
            "delta": float(raw_delta) if raw_delta is not None else None,
            "gamma": float(raw_gamma) if raw_gamma is not None else None,
            "theta": float(raw_theta) if raw_theta is not None else None,
            "vega": float(raw_vega) if raw_vega is not None else None,
            "quote_timestamp": quote_ts,
            "greeks_timestamp": greeks_ts,
        }
