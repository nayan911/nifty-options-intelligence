"""Rate-limited, asynchronous REST Client for FYERS Data API v3."""

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from fyers_apiv3 import fyersModel

from nifty_engine.core.config import settings
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
        options_chain = chain_data.get("optionsChain", [])

        # Parse strikes
        parsed_strikes: List[OptionChainStrike] = []
        total_call_oi = 0
        total_put_oi = 0

        for item in options_chain:
            strike_price = float(item.get("strike_price", 0))
            call_dict = {}
            put_dict = {}

            # Parse call
            if "ce" in item and item["ce"]:
                ce = item["ce"]
                call_oi = int(ce.get("oi", 0))
                total_call_oi += call_oi
                greeks = ce.get("greeks", {})
                call_dict = {
                    "ltp": float(ce.get("ltp", 0)),
                    "oi": call_oi,
                    "volume": int(ce.get("volume", 0)),
                    "iv": float(greeks.get("iv", 0)),
                    "delta": float(greeks.get("delta", 0)),
                    "gamma": float(greeks.get("gamma", 0)),
                    "theta": float(greeks.get("theta", 0)),
                    "vega": float(greeks.get("vega", 0)),
                }

            # Parse put
            if "pe" in item and item["pe"]:
                pe = item["pe"]
                put_oi = int(pe.get("oi", 0))
                total_put_oi += put_oi
                greeks = pe.get("greeks", {})
                put_dict = {
                    "ltp": float(pe.get("ltp", 0)),
                    "oi": put_oi,
                    "volume": int(pe.get("volume", 0)),
                    "iv": float(greeks.get("iv", 0)),
                    "delta": float(greeks.get("delta", 0)),
                    "gamma": float(greeks.get("gamma", 0)),
                    "theta": float(greeks.get("theta", 0)),
                    "vega": float(greeks.get("vega", 0)),
                }

            parsed_strikes.append(
                OptionChainStrike(
                    strike=strike_price,
                    call=call_dict or None,
                    put=put_dict or None,
                )
            )

        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0.0

        return OptionChainSnapshot(
            underlying="NIFTY" if "NIFTY" in symbol else symbol,
            spot_price=float(chain_data.get("underlyingValue", 0)),
            timestamp=datetime.now(),
            expiry=date.today(),
            strikes=parsed_strikes,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr=pcr,
        )
