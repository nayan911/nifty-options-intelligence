"""Rate-limited, asynchronous REST Client for FYERS Data API v3."""

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytz
from fyers_apiv3 import fyersModel

IST = pytz.timezone("Asia/Kolkata")

from nifty_engine.core.interfaces import (
    Candle,
    OptionChainSnapshot,
    OptionChainStrike,
)
from nifty_engine.core.rate_limiter import AsyncTokenBucketRateLimiter
from nifty_engine.data.fyers.auth import FyersAuthProvider


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
    ) -> List[Candle]:
        """Fetch historical candles with rate limiting.

        resolution: '1', '5', '15', 'D'
        """
        await self.rate_limiter.acquire()

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": date_from,
            "range_to": date_to,
            "cont_flag": "1",
        }

        # Run SDK synchronous call in async thread pool
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.history(data=data))

        if not response or response.get("s") != "ok":
            error_msg = response.get("message", "Error fetching historical data")
            raise RuntimeError(f"FYERS Historical Data Error ({symbol}): {error_msg}")

        raw_candles = response.get("candles", [])
        candles = []
        for c in raw_candles:
            # c = [epoch_timestamp, open, high, low, close, volume]
            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=f"{resolution}m" if resolution.isdigit() else resolution,
                    timestamp=datetime.fromtimestamp(c[0]),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=int(c[5]),
                    is_closed=True,
                )
            )
        return candles

    async def get_option_chain(
        self,
        symbol: str = "NSE:NIFTY50-INDEX",
        strikecount: int = 10,
    ) -> OptionChainSnapshot:
        """Fetch live option chain with Greeks."""
        await self.rate_limiter.acquire()

        data = {
            "symbol": symbol,
            "strikecount": strikecount,
            "timestamp": "",
            "greeks": "1",
        }

        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.optionchain(data=data))

        if not response or response.get("s") != "ok":
            error_msg = response.get("message", "Error fetching option chain")
            raise RuntimeError(f"FYERS Option Chain Error ({symbol}): {error_msg}")

        chain_data = response.get("data", {})
        return self.parse_option_chain_payload(chain_data, underlying_symbol=symbol)

    @staticmethod
    def parse_option_chain_payload(chain_data: Dict[str, Any], underlying_symbol: str) -> OptionChainSnapshot:
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

        # Check top-level expiryData (official FYERS v3 specification)
        expiry_data = chain_data.get("expiryData", [])
        if expiry_data and isinstance(expiry_data, list):
            first_exp = expiry_data[0]
            if isinstance(first_exp, dict):
                if first_exp.get("timestamp"):
                    detected_expiry = datetime.fromtimestamp(first_exp["timestamp"], tz=IST).date()
                elif first_exp.get("date"):
                    try:
                        detected_expiry = datetime.strptime(first_exp["date"], "%d-%b-%Y").date()
                    except ValueError:
                        pass
            elif isinstance(first_exp, (int, float)):
                detected_expiry = datetime.fromtimestamp(first_exp, tz=IST).date()

        for item in options_chain:
            # Case 1: Nested structure (item contains 'ce' and/or 'pe')
            if "ce" in item or "pe" in item:
                strike_p = float(item.get("strike_price", 0.0))
                if strike_p not in strike_map:
                    strike_map[strike_p] = {"call": None, "put": None}

                if item.get("ce"):
                    c = item["ce"]
                    oi = int(c.get("oi", 0))
                    total_call_oi += oi
                    strike_map[strike_p]["call"] = FyersRestClient._extract_contract_fields(c)
                if item.get("pe"):
                    p = item["pe"]
                    oi = int(p.get("oi", 0))
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
                oi = int(item.get("oi", 0))

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

        return OptionChainSnapshot(
            underlying=underlying,
            spot_price=spot_price,
            api_timestamp=datetime.now(IST),
            expiry=detected_expiry,
            strikes=parsed_strikes,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr=pcr,
        )

    @staticmethod
    def _extract_contract_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts standard contract fields including Greeks, OI change (oich), volume, and quotes."""
        greeks = raw.get("greeks", {}) or {}

        # Parse documented FYERS oich / oichange / oi_change
        raw_oich = raw.get("oich")
        if raw_oich is None:
            raw_oich = raw.get("oichange")
        if raw_oich is None:
            raw_oich = raw.get("oi_change", 0)

        return {
            "symbol": raw.get("symbol", ""),
            "ltp": float(raw.get("ltp", 0.0)),
            "oi": int(raw.get("oi", 0)),
            "oi_change": int(raw_oich),
            "volume": int(raw.get("volume", raw.get("vol_traded_today", 0))),
            "bid": float(raw.get("bid", 0.0)) if raw.get("bid") is not None else None,
            "ask": float(raw.get("ask", 0.0)) if raw.get("ask") is not None else None,
            "iv": float(greeks.get("iv", raw.get("iv", 0.0))),
            "delta": float(greeks.get("delta", raw.get("delta", 0.0))),
            "gamma": float(greeks.get("gamma", raw.get("gamma", 0.0))),
            "theta": float(greeks.get("theta", raw.get("theta", 0.0))),
            "vega": float(greeks.get("vega", raw.get("vega", 0.0))),
        }
