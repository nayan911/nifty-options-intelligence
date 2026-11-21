"""Open Interest (OI) Analysis Engine for NIFTY Derivatives.

Provides strike-level options and futures OI flow analysis, buildup classification
with proper directional semantics (CE vs PE), dynamic Call/Put walls, Max Pain,
and multi-window timestamped comparisons.
"""

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import pytz

from nifty_engine.core.interfaces import Candle, OptionChainSnapshot

IST = pytz.timezone("Asia/Kolkata")


class BuildupType(str, Enum):
    """Positioning heuristic based on price and OI change."""

    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"
    NEUTRAL = "NEUTRAL"


class DirectionalBias(str, Enum):
    """Directional bias implied by the positioning heuristic."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class StrikeOIAnalysis:
    """Detailed OI and buildup analysis for a specific strike price."""

    strike: float
    # Call contract metrics
    call_oi: int = 0
    call_oi_change: int = 0
    call_oi_change_pct: float = 0.0
    call_ltp: float = 0.0
    call_ltp_change: float = 0.0
    call_volume: int = 0
    call_volume_change: int = 0
    call_buildup: BuildupType = BuildupType.NEUTRAL
    call_directional_bias: DirectionalBias = DirectionalBias.NEUTRAL

    # Put contract metrics
    put_oi: int = 0
    put_oi_change: int = 0
    put_oi_change_pct: float = 0.0
    put_ltp: float = 0.0
    put_ltp_change: float = 0.0
    put_volume: int = 0
    put_volume_change: int = 0
    put_buildup: BuildupType = BuildupType.NEUTRAL
    put_directional_bias: DirectionalBias = DirectionalBias.NEUTRAL

    # Net bias across CE and PE for this strike
    net_strike_bias: DirectionalBias = DirectionalBias.NEUTRAL


@dataclass
class FuturesOIAnalysis:
    """Futures price and OI positioning analysis."""

    symbol: str
    timestamp: datetime
    price: float
    price_change: float
    price_change_pct: float
    oi: int
    oi_change: int
    oi_change_pct: float
    buildup: BuildupType
    directional_bias: DirectionalBias


@dataclass
class OIProfile:
    """Consolidated Open Interest profile for an option chain snapshot."""

    timestamp: datetime
    expiry: date
    spot_price: float
    call_wall_strike: float
    put_wall_strike: float
    call_wall_oi: int
    put_wall_oi: int
    max_pain_strike: Optional[float]
    strikes: Dict[float, StrikeOIAnalysis]
    near_atm_strikes: List[StrikeOIAnalysis]
    aggregate_bias: DirectionalBias
    futures_analysis: Optional[FuturesOIAnalysis] = None
    observation_interval_seconds: float = 0.0
    is_stale: bool = False


class OpenInterestAnalyzer:
    """Ingests option chain snapshots and futures candles to produce causal,

    statistically bounded Open Interest flow analysis.
    """

    def __init__(
        self,
        min_oi_change_threshold_pct: float = 2.0,
        min_oi_change_contracts: int = 500,
        min_price_change_pct: float = 0.5,
        max_snapshot_age_seconds: float = 360.0,
        near_atm_strikes_count: int = 5,
        history_max_len: int = 50,
    ):
        self.min_oi_change_threshold_pct = min_oi_change_threshold_pct
        self.min_oi_change_contracts = min_oi_change_contracts
        self.min_price_change_pct = min_price_change_pct
        self.max_snapshot_age_seconds = max_snapshot_age_seconds
        self.near_atm_strikes_count = near_atm_strikes_count
        self.history_max_len = history_max_len

        # Snapshot history partitioned by expiry date to guarantee expiry isolation
        self._chain_history: Dict[date, Deque[OptionChainSnapshot]] = {}
        # Latest profile per expiry date
        self.latest_profiles: Dict[date, OIProfile] = {}

        # Futures tracking: rolling history of (timestamp, price, oi)
        self._futures_history: Deque[Tuple[datetime, float, int]] = deque(maxlen=history_max_len)
        self.latest_futures_analysis: Optional[FuturesOIAnalysis] = None

    def on_futures_candle(self, candle: Candle) -> Optional[FuturesOIAnalysis]:
        """Updates futures OI and price state from a newly closed candle."""
        if candle.oi is None or candle.close <= 0.0:
            return self.latest_futures_analysis

        current_time = candle.timestamp
        current_price = candle.close
        current_oi = candle.oi

        if not self._futures_history:
            self._futures_history.append((current_time, current_price, current_oi))
            self.latest_futures_analysis = FuturesOIAnalysis(
                symbol=candle.symbol,
                timestamp=current_time,
                price=current_price,
                price_change=0.0,
                price_change_pct=0.0,
                oi=current_oi,
                oi_change=0,
                oi_change_pct=0.0,
                buildup=BuildupType.NEUTRAL,
                directional_bias=DirectionalBias.NEUTRAL,
            )
            return self.latest_futures_analysis

        prev_time, prev_price, prev_oi = self._futures_history[-1]
        self._futures_history.append((current_time, current_price, current_oi))

        price_diff = current_price - prev_price
        price_diff_pct = (price_diff / prev_price * 100.0) if prev_price > 0 else 0.0
        oi_diff = current_oi - prev_oi
        oi_diff_pct = (oi_diff / prev_oi * 100.0) if prev_oi > 0 else 0.0

        buildup, bias = self._classify_buildup(
            price_change=price_diff,
            price_change_pct=price_diff_pct,
            oi_change=oi_diff,
            oi_change_pct=oi_diff_pct,
            instrument_type="FUT",
        )

        self.latest_futures_analysis = FuturesOIAnalysis(
            symbol=candle.symbol,
            timestamp=current_time,
            price=current_price,
            price_change=round(price_diff, 2),
            price_change_pct=round(price_diff_pct, 2),
            oi=current_oi,
            oi_change=oi_diff,
            oi_change_pct=round(oi_diff_pct, 2),
            buildup=buildup,
            directional_bias=bias,
        )
        return self.latest_futures_analysis

    def on_option_chain(
        self,
        snapshot: OptionChainSnapshot,
        as_of: Optional[datetime] = None,
    ) -> OIProfile:
        """Processes an incoming OptionChainSnapshot.

        Calculates actual interval changes between consecutive snapshots for the same expiry,
        classifies directional CE/PE buildup, and identifies Call/Put walls.
        """
        expiry = snapshot.expiry
        if expiry not in self._chain_history:
            self._chain_history[expiry] = deque(maxlen=self.history_max_len)

        history = self._chain_history[expiry]
        prev_snapshot: Optional[OptionChainSnapshot] = history[-1] if history else None
        history.append(snapshot)

        current_time = as_of or snapshot.api_timestamp
        interval_seconds = 0.0
        if prev_snapshot is not None:
            interval_seconds = max(0.0, (snapshot.api_timestamp - prev_snapshot.api_timestamp).total_seconds())

        is_stale = False
        if (current_time - snapshot.api_timestamp).total_seconds() > self.max_snapshot_age_seconds:
            is_stale = True

        # Build strike map of previous snapshot for exact delta calculations
        prev_strikes_map = {}
        if prev_snapshot is not None:
            for s in prev_snapshot.strikes:
                prev_strikes_map[s.strike] = s

        strikes_analysis: Dict[float, StrikeOIAnalysis] = {}
        max_call_oi = -1
        call_wall_strike = 0.0
        max_put_oi = -1
        put_wall_strike = 0.0

        bullish_count = 0
        bearish_count = 0

        for s in snapshot.strikes:
            strike = s.strike
            prev_s = prev_strikes_map.get(strike)

            # Call analysis
            c_curr = s.call or {}
            c_prev = (prev_s.call or {}) if prev_s else {}
            c_oi = int(c_curr.get("oi", 0))
            c_ltp = float(c_curr.get("ltp", 0.0))
            c_vol = int(c_curr.get("volume", 0))

            c_prev_oi = int(c_prev.get("oi", c_oi))
            c_prev_ltp = float(c_prev.get("ltp", c_ltp))
            c_prev_vol = int(c_prev.get("volume", c_vol))

            c_oi_change = c_oi - c_prev_oi
            c_oi_change_pct = (c_oi_change / c_prev_oi * 100.0) if c_prev_oi > 0 else 0.0
            c_ltp_change = c_ltp - c_prev_ltp
            c_ltp_change_pct = (c_ltp_change / c_prev_ltp * 100.0) if c_prev_ltp > 0 else 0.0
            c_vol_change = c_vol - c_prev_vol

            c_buildup, c_bias = self._classify_buildup(
                price_change=c_ltp_change,
                price_change_pct=c_ltp_change_pct,
                oi_change=c_oi_change,
                oi_change_pct=c_oi_change_pct,
                instrument_type="CE",
            )

            if c_oi > max_call_oi:
                max_call_oi = c_oi
                call_wall_strike = strike

            # Put analysis
            p_curr = s.put or {}
            p_prev = (prev_s.put or {}) if prev_s else {}
            p_oi = int(p_curr.get("oi", 0))
            p_ltp = float(p_curr.get("ltp", 0.0))
            p_vol = int(p_curr.get("volume", 0))

            p_prev_oi = int(p_prev.get("oi", p_oi))
            p_prev_ltp = float(p_prev.get("ltp", p_ltp))
            p_prev_vol = int(p_prev.get("volume", p_vol))

            p_oi_change = p_oi - p_prev_oi
            p_oi_change_pct = (p_oi_change / p_prev_oi * 100.0) if p_prev_oi > 0 else 0.0
            p_ltp_change = p_ltp - p_prev_ltp
            p_ltp_change_pct = (p_ltp_change / p_prev_ltp * 100.0) if p_prev_ltp > 0 else 0.0
            p_vol_change = p_vol - p_prev_vol

            p_buildup, p_bias = self._classify_buildup(
                price_change=p_ltp_change,
                price_change_pct=p_ltp_change_pct,
                oi_change=p_oi_change,
                oi_change_pct=p_oi_change_pct,
                instrument_type="PE",
            )

            if p_oi > max_put_oi:
                max_put_oi = p_oi
                put_wall_strike = strike

            # Net strike bias
            net_bias = DirectionalBias.NEUTRAL
            if c_bias == DirectionalBias.BULLISH and p_bias in (DirectionalBias.BULLISH, DirectionalBias.NEUTRAL):
                net_bias = DirectionalBias.BULLISH
            elif c_bias == DirectionalBias.BEARISH and p_bias in (DirectionalBias.BEARISH, DirectionalBias.NEUTRAL):
                net_bias = DirectionalBias.BEARISH
            elif p_bias == DirectionalBias.BULLISH and c_bias == DirectionalBias.NEUTRAL:
                net_bias = DirectionalBias.BULLISH
            elif p_bias == DirectionalBias.BEARISH and c_bias == DirectionalBias.NEUTRAL:
                net_bias = DirectionalBias.BEARISH

            if net_bias == DirectionalBias.BULLISH:
                bullish_count += 1
            elif net_bias == DirectionalBias.BEARISH:
                bearish_count += 1

            strikes_analysis[strike] = StrikeOIAnalysis(
                strike=strike,
                call_oi=c_oi,
                call_oi_change=c_oi_change,
                call_oi_change_pct=round(c_oi_change_pct, 2),
                call_ltp=c_ltp,
                call_ltp_change=round(c_ltp_change, 2),
                call_volume=c_vol,
                call_volume_change=c_vol_change,
                call_buildup=c_buildup,
                call_directional_bias=c_bias,
                put_oi=p_oi,
                put_oi_change=p_oi_change,
                put_oi_change_pct=round(p_oi_change_pct, 2),
                put_ltp=p_ltp,
                put_ltp_change=round(p_ltp_change, 2),
                put_volume=p_vol,
                put_volume_change=p_vol_change,
                put_buildup=p_buildup,
                put_directional_bias=p_bias,
                net_strike_bias=net_bias,
            )

        # Near-ATM strikes extraction
        sorted_strikes = sorted(strikes_analysis.keys())
        near_atm: List[StrikeOIAnalysis] = []
        if sorted_strikes:
            spot = snapshot.spot_price
            sorted_by_dist = sorted(sorted_strikes, key=lambda k: abs(k - spot))
            atm_strikes = sorted(sorted_by_dist[: 2 * self.near_atm_strikes_count + 1])
            near_atm = [strikes_analysis[k] for k in atm_strikes]

        # Aggregate bias across strikes
        aggregate_bias = DirectionalBias.NEUTRAL
        if near_atm:
            near_bull = sum(1 for s in near_atm if s.net_strike_bias == DirectionalBias.BULLISH)
            near_bear = sum(1 for s in near_atm if s.net_strike_bias == DirectionalBias.BEARISH)
            if near_bull > near_bear and near_bull >= 2:
                aggregate_bias = DirectionalBias.BULLISH
            elif near_bear > near_bull and near_bear >= 2:
                aggregate_bias = DirectionalBias.BEARISH

        # Optional Max Pain calculation
        max_pain = self._calculate_max_pain(snapshot)

        profile = OIProfile(
            timestamp=snapshot.api_timestamp,
            expiry=expiry,
            spot_price=snapshot.spot_price,
            call_wall_strike=call_wall_strike,
            put_wall_strike=put_wall_strike,
            call_wall_oi=max(0, max_call_oi),
            put_wall_oi=max(0, max_put_oi),
            max_pain_strike=max_pain,
            strikes=strikes_analysis,
            near_atm_strikes=near_atm,
            aggregate_bias=aggregate_bias,
            futures_analysis=self.latest_futures_analysis,
            observation_interval_seconds=interval_seconds,
            is_stale=is_stale,
        )
        self.latest_profiles[expiry] = profile
        return profile

    def analyze_concentration_near_level(
        self,
        level: float,
        expiry: Optional[date] = None,
        tolerance_points: float = 50.0,
    ) -> Dict[str, Any]:
        """Analyzes Call vs Put OI accumulation and net flow near a specific technical level.

        Useful for checking whether Call writers are aggressively defending a resistance level
        or Put writers are supporting a breakout.
        """
        profile = self.get_latest_profile(expiry)
        if not profile or not profile.strikes:
            return {
                "level": level,
                "total_call_oi": 0,
                "total_put_oi": 0,
                "call_oi_change": 0,
                "put_oi_change": 0,
                "call_writers_active": False,
                "put_writers_active": False,
                "dominant_bias": DirectionalBias.NEUTRAL,
                "is_available": False,
            }

        matching = [
            s for k, s in profile.strikes.items() if abs(k - level) <= tolerance_points
        ]

        if not matching:
            return {
                "level": level,
                "total_call_oi": 0,
                "total_put_oi": 0,
                "call_oi_change": 0,
                "put_oi_change": 0,
                "call_writers_active": False,
                "put_writers_active": False,
                "dominant_bias": DirectionalBias.NEUTRAL,
                "is_available": False,
            }

        total_call_oi = sum(s.call_oi for s in matching)
        total_put_oi = sum(s.put_oi for s in matching)
        call_oi_change = sum(s.call_oi_change for s in matching)
        put_oi_change = sum(s.put_oi_change for s in matching)

        # Call writers are active if Call OI increased with Price falling or flat
        call_writers_active = (
            call_oi_change >= self.min_oi_change_contracts
            and any(s.call_buildup == BuildupType.SHORT_BUILDUP for s in matching)
        )
        # Put writers are active if Put OI increased with Put price falling (support writing)
        put_writers_active = (
            put_oi_change >= self.min_oi_change_contracts
            and any(s.put_buildup == BuildupType.SHORT_BUILDUP for s in matching)
        )

        dominant_bias = DirectionalBias.NEUTRAL
        if put_writers_active and not call_writers_active:
            dominant_bias = DirectionalBias.BULLISH
        elif call_writers_active and not put_writers_active:
            dominant_bias = DirectionalBias.BEARISH

        return {
            "level": level,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "call_oi_change": call_oi_change,
            "put_oi_change": put_oi_change,
            "call_writers_active": call_writers_active,
            "put_writers_active": put_writers_active,
            "dominant_bias": dominant_bias,
            "is_available": True,
        }

    def get_latest_profile(self, expiry: Optional[date] = None) -> Optional[OIProfile]:
        """Returns the latest profile, optionally filtered by expiry date."""
        if expiry is not None:
            return self.latest_profiles.get(expiry)
        if not self.latest_profiles:
            return None
        # Return nearest expiry profile
        sorted_expiries = sorted(self.latest_profiles.keys())
        return self.latest_profiles[sorted_expiries[0]]

    def _classify_buildup(
        self,
        price_change: float,
        price_change_pct: float,
        oi_change: int,
        oi_change_pct: float,
        instrument_type: str,  # "FUT", "CE", "PE"
    ) -> Tuple[BuildupType, DirectionalBias]:
        """Classifies positioning heuristic and assigns strict directional bias per instrument type.

        CE:
        - Price UP, OI UP   -> Long Buildup  -> BULLISH (Call Buying)
        - Price DOWN, OI UP -> Short Buildup -> BEARISH (Call Writing / Resistance)
        - Price UP, OI DOWN -> Short Covering -> BULLISH (Call Short Covering)
        - Price DOWN, OI DOWN -> Long Unwinding -> BEARISH (Call Long Liquidation)

        PE:
        - Price UP, OI UP   -> Long Buildup  -> BEARISH (Put Buying / Hedging)
        - Price DOWN, OI UP -> Short Buildup -> BULLISH (Put Writing / Support)
        - Price UP, OI DOWN -> Short Covering -> BEARISH (Put Short Covering)
        - Price DOWN, OI DOWN -> Long Unwinding -> BULLISH (Put Long Liquidation)

        FUT:
        - Price UP, OI UP   -> Long Buildup  -> BULLISH
        - Price DOWN, OI UP -> Short Buildup -> BEARISH
        - Price UP, OI DOWN -> Short Covering -> BULLISH
        - Price DOWN, OI DOWN -> Long Unwinding -> BEARISH
        """
        # Statistical filter: ignore trivial fluctuations below significance thresholds
        is_oi_significant = (
            abs(oi_change) >= self.min_oi_change_contracts
            and abs(oi_change_pct) >= self.min_oi_change_threshold_pct
        )
        is_price_significant = abs(price_change_pct) >= self.min_price_change_pct

        if not is_oi_significant and not is_price_significant:
            return BuildupType.NEUTRAL, DirectionalBias.NEUTRAL

        price_up = price_change > 0
        oi_up = oi_change > 0

        if price_up and oi_up:
            buildup = BuildupType.LONG_BUILDUP
        elif not price_up and oi_up:
            buildup = BuildupType.SHORT_BUILDUP
        elif price_up and not oi_up:
            buildup = BuildupType.SHORT_COVERING
        else:
            buildup = BuildupType.LONG_UNWINDING

        # Directional mapping per instrument type
        if instrument_type in ("FUT", "CE"):
            if buildup in (BuildupType.LONG_BUILDUP, BuildupType.SHORT_COVERING):
                bias = DirectionalBias.BULLISH
            else:
                bias = DirectionalBias.BEARISH
        elif instrument_type == "PE":
            # For Puts, price down + OI up is Put writing (Bullish); price up + OI up is Put buying (Bearish)
            if buildup in (BuildupType.SHORT_BUILDUP, BuildupType.LONG_UNWINDING):
                bias = DirectionalBias.BULLISH
            else:
                bias = DirectionalBias.BEARISH
        else:
            bias = DirectionalBias.NEUTRAL

        return buildup, bias

    def _calculate_max_pain(self, snapshot: OptionChainSnapshot) -> Optional[float]:
        """Calculates optional descriptive Max Pain strike.

        Total payout to option holders at settlement price K is:
        Sum(Call_OI * max(0, K - Strike)) + Sum(Put_OI * max(0, Strike - K))
        The strike K with the lowest total payout is Max Pain.
        """
        if not snapshot.strikes:
            return None

        strikes = [s.strike for s in snapshot.strikes]
        min_total_payout = float("inf")
        max_pain_strike = None

        for k in strikes:
            total_payout = 0.0
            for s in snapshot.strikes:
                strike_price = s.strike
                call_oi = (s.call or {}).get("oi", 0)
                put_oi = (s.put or {}).get("oi", 0)

                # Call intrinsic value at settlement K
                if k > strike_price:
                    total_payout += call_oi * (k - strike_price)
                # Put intrinsic value at settlement K
                if k < strike_price:
                    total_payout += put_oi * (strike_price - k)

            if total_payout < min_total_payout:
                min_total_payout = total_payout
                max_pain_strike = k

        return max_pain_strike
