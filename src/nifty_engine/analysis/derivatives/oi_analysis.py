"""Open Interest (OI) Analysis Engine for NIFTY Derivatives.

Provides strike-level options and futures OI flow analysis, buildup classification
with proper directional semantics (CE vs PE), dynamic Call/Put walls, Max Pain,
and multi-window timestamped comparisons with strict symbol, timeframe, and expiry isolation.
"""

from collections import deque
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import pytz

from nifty_engine.core.calendar import get_bar_close_time
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
    timeframe: str
    timestamp: datetime
    price: float
    price_change: float
    price_change_pct: float
    oi: int
    oi_change: int
    oi_change_pct: float
    buildup: BuildupType
    directional_bias: DirectionalBias
    bar_close_time: Optional[datetime] = None
    available_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.bar_close_time is None:
            self.bar_close_time = self.timestamp
        if self.available_at is None:
            self.available_at = self.bar_close_time


@dataclass
class OIProfile:
    """Consolidated Open Interest profile for an option chain snapshot."""

    underlying: str
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
    observation_interval_seconds: float = 0.0
    is_stale: bool = False


class OpenInterestAnalyzer:
    """Ingests option chain snapshots and futures candles to produce causal,

    statistically bounded Open Interest flow analysis.
    """

    def __init__(
        self,
        # Option thresholds
        min_opt_oi_change_threshold_pct: float = 1.5,
        min_opt_oi_change_contracts: int = 300,
        min_opt_price_change_points: float = 1.0,
        min_opt_price_change_pct: float = 1.5,
        # Futures thresholds (points and percentage tailored to underlying index futures)
        min_fut_oi_change_threshold_pct: float = 0.1,
        min_fut_oi_change_contracts: int = 150,
        min_fut_price_change_points: float = 3.0,
        min_fut_price_change_pct: float = 0.02,
        # Freshness and observation limits
        max_snapshot_age_seconds: float = 360.0,
        max_futures_age_seconds: float = 600.0,
        near_atm_strikes_count: int = 5,
        history_max_len: int = 50,
        # Backward compatibility aliases
        min_oi_change_threshold_pct: Optional[float] = None,
        min_oi_change_contracts: Optional[int] = None,
        min_price_change_pct: Optional[float] = None,
    ):
        if min_oi_change_threshold_pct is not None:
            min_opt_oi_change_threshold_pct = min_oi_change_threshold_pct
            min_fut_oi_change_threshold_pct = min_oi_change_threshold_pct
        if min_oi_change_contracts is not None:
            min_opt_oi_change_contracts = min_oi_change_contracts
            min_fut_oi_change_contracts = min_oi_change_contracts
        if min_price_change_pct is not None:
            min_opt_price_change_pct = min_price_change_pct
            min_fut_price_change_pct = min_price_change_pct

        self.min_opt_oi_change_threshold_pct = min_opt_oi_change_threshold_pct
        self.min_opt_oi_change_contracts = min_opt_oi_change_contracts
        self.min_opt_price_change_points = min_opt_price_change_points
        self.min_opt_price_change_pct = min_opt_price_change_pct

        self.min_fut_oi_change_threshold_pct = min_fut_oi_change_threshold_pct
        self.min_fut_oi_change_contracts = min_fut_oi_change_contracts
        self.min_fut_price_change_points = min_fut_price_change_points
        self.min_fut_price_change_pct = min_fut_price_change_pct

        self.max_snapshot_age_seconds = max_snapshot_age_seconds
        self.max_futures_age_seconds = max_futures_age_seconds
        self.near_atm_strikes_count = near_atm_strikes_count
        self.history_max_len = history_max_len

        # Current trading session date
        self.current_session_date: Optional[date] = None

        # Snapshot & Profile history partitioned by (underlying, expiry)
        self._chain_history: Dict[Tuple[str, date], Deque[OptionChainSnapshot]] = {}
        self._profile_history: Dict[Tuple[str, date], Deque[OIProfile]] = {}
        self.latest_profiles: Dict[Tuple[str, date], OIProfile] = {}

        # Futures tracking partitioned strictly by (symbol, timeframe)
        self._futures_history: Dict[Tuple[str, str], Deque[FuturesOIAnalysis]] = {}
        self.latest_futures_analysis: Dict[Tuple[str, str], FuturesOIAnalysis] = {}

    def on_futures_candle(self, candle: Candle) -> Optional[FuturesOIAnalysis]:
        """Updates futures OI and price state from a newly closed candle with symbol and timeframe isolation."""
        if not candle.is_closed or candle.oi is None or candle.close <= 0.0:
            return None

        key = (candle.symbol, candle.timeframe)
        if key not in self._futures_history:
            self._futures_history[key] = deque(maxlen=self.history_max_len)

        history = self._futures_history[key]
        current_time = candle.timestamp
        current_price = candle.close
        current_oi = candle.oi

        close_time = get_bar_close_time(candle.timestamp, candle.timeframe)
        avail_at = max(candle.observed_at, close_time) if candle.observed_at is not None else close_time

        # Reject candles from prior trading sessions
        if self.current_session_date is not None and candle.timestamp.date() < self.current_session_date:
            return history[-1] if history else None

        # Reject duplicate or out-of-order candles: opening timestamp must increase strictly
        if history and candle.timestamp <= history[-1].timestamp:
            return history[-1]

        # Session rollover tracking: clean baseline for a new trading day
        is_new_session = (
            self.current_session_date is None
            or candle.timestamp.date() > self.current_session_date
            or (history and history[-1].bar_close_time.date() < close_time.date())
        )
        if self.current_session_date is None or candle.timestamp.date() > self.current_session_date:
            self.current_session_date = candle.timestamp.date()

        if not history or is_new_session:
            analysis = FuturesOIAnalysis(
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                timestamp=current_time,
                bar_close_time=close_time,
                available_at=avail_at,
                price=current_price,
                price_change=0.0,
                price_change_pct=0.0,
                oi=current_oi,
                oi_change=0,
                oi_change_pct=0.0,
                buildup=BuildupType.NEUTRAL,
                directional_bias=DirectionalBias.NEUTRAL,
            )
            history.append(analysis)
            self.latest_futures_analysis[key] = analysis
            return analysis

        prev = history[-1]
        price_diff = current_price - prev.price
        price_diff_pct = (price_diff / prev.price * 100.0) if prev.price > 0 else 0.0
        oi_diff = current_oi - prev.oi
        oi_diff_pct = (oi_diff / prev.oi * 100.0) if prev.oi > 0 else 0.0

        buildup, bias = self._classify_buildup(
            price_change=price_diff,
            price_change_pct=price_diff_pct,
            oi_change=oi_diff,
            oi_change_pct=oi_diff_pct,
            instrument_type="FUT",
        )

        analysis = FuturesOIAnalysis(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            timestamp=current_time,
            bar_close_time=close_time,
            available_at=avail_at,
            price=current_price,
            price_change=round(price_diff, 2),
            price_change_pct=round(price_diff_pct, 2),
            oi=current_oi,
            oi_change=oi_diff,
            oi_change_pct=round(oi_diff_pct, 2),
            buildup=buildup,
            directional_bias=bias,
        )
        history.append(analysis)
        self.latest_futures_analysis[key] = analysis
        return analysis

    def get_futures_analysis(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "5m",
        as_of: Optional[datetime] = None,
    ) -> Optional[FuturesOIAnalysis]:
        """Retrieves causal futures positioning analysis as of a specific cutoff timestamp."""
        # If symbol not specified, find matching primary symbol in history
        matching_keys = [k for k in self._futures_history if (symbol is None or k[0] == symbol) and k[1] == timeframe]
        if not matching_keys:
            # Fall back to any timeframe (e.g. 1m) if requested timeframe (e.g. 5m) is not yet populated
            matching_keys = [k for k in self._futures_history if (symbol is None or k[0] == symbol)]
            if not matching_keys:
                return None

        key = matching_keys[0]
        history = self._futures_history[key]
        if not history:
            return None

        if as_of is None:
            return history[-1]

        # Reverse lookup for most recent observation available on or before as_of
        for obs in reversed(history):
            if obs.available_at <= as_of:
                age = (as_of - obs.available_at).total_seconds()
                if age <= self.max_futures_age_seconds:
                    return obs
                return None

        return None

    def on_option_chain(
        self,
        snapshot: OptionChainSnapshot,
        as_of: Optional[datetime] = None,
    ) -> Optional[OIProfile]:
        """Processes an incoming OptionChainSnapshot with underlying and expiry isolation.

        Calculates actual interval changes between consecutive snapshots for the same contract set,
        classifies directional CE/PE buildup, and identifies Call/Put walls.
        """
        underlying = snapshot.underlying.upper()
        expiry = snapshot.expiry
        current_time = as_of or snapshot.api_timestamp

        key = (underlying, expiry)
        if key not in self._chain_history:
            self._chain_history[key] = deque(maxlen=self.history_max_len)
            self._profile_history[key] = deque(maxlen=self.history_max_len)

        history = self._chain_history[key]

        # Session date update
        is_new_session = (
            self.current_session_date is None
            or current_time.date() > self.current_session_date
            or (bool(history) and history[-1].api_timestamp.date() < current_time.date())
        )
        if self.current_session_date is None or current_time.date() > self.current_session_date:
            self.current_session_date = current_time.date()

        # Ignore expired contracts
        if expiry < self.current_session_date:
            return None

        # Reject duplicate or out-of-order snapshots
        if history and snapshot.api_timestamp <= history[-1].api_timestamp:
            return self._profile_history[key][-1] if self._profile_history[key] else None

        prev_snapshot: Optional[OptionChainSnapshot] = None
        if history and not is_new_session:
            prev_snapshot = history[-1]
        history.append(snapshot)

        interval_seconds = 0.0
        if prev_snapshot is not None:
            interval_seconds = max(0.0, (snapshot.api_timestamp - prev_snapshot.api_timestamp).total_seconds())

        is_stale = (current_time - snapshot.api_timestamp).total_seconds() > self.max_snapshot_age_seconds

        # Map previous strikes for exact delta calculations
        prev_strikes_map = {s.strike: s for s in prev_snapshot.strikes} if prev_snapshot is not None else {}

        strikes_analysis: Dict[float, StrikeOIAnalysis] = {}
        max_call_oi = -1
        call_wall_strike = 0.0
        max_put_oi = -1
        put_wall_strike = 0.0

        for s in snapshot.strikes:
            strike = s.strike
            prev_s = prev_strikes_map.get(strike)

            # Call analysis
            c_curr = s.call
            c_prev = prev_s.call if prev_s else None

            if c_curr is not None:
                c_oi_raw = c_curr.get("oi")
                c_ltp_raw = c_curr.get("ltp")
                c_vol_raw = c_curr.get("volume")

                c_oi = int(c_oi_raw) if c_oi_raw is not None else 0
                c_ltp = float(c_ltp_raw) if c_ltp_raw is not None else 0.0
                c_vol = int(c_vol_raw) if c_vol_raw is not None else 0

                if (
                    c_prev is not None
                    and c_oi_raw is not None
                    and c_prev.get("oi") is not None
                    and c_ltp_raw is not None
                    and c_prev.get("ltp") is not None
                ):
                    c_prev_oi = int(c_prev.get("oi", c_oi))
                    c_prev_ltp = float(c_prev.get("ltp", c_ltp))
                    c_prev_vol = int(c_prev.get("volume", c_vol))

                    c_oi_change = c_oi - c_prev_oi
                    c_oi_change_pct = (c_oi_change / c_prev_oi * 100.0) if c_prev_oi > 0 else 0.0
                    c_ltp_change = c_ltp - c_prev_ltp
                    c_ltp_change_pct = (c_ltp_change / c_prev_ltp * 100.0) if c_prev_ltp > 0 else 0.0
                    c_vol_change = c_vol - c_prev_vol
                else:
                    # Newly appearing strike or missing data: establish baseline, zero delta
                    c_oi_change = 0
                    c_oi_change_pct = 0.0
                    c_ltp_change = 0.0
                    c_ltp_change_pct = 0.0
                    c_vol_change = 0

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
            else:
                c_oi = 0
                c_oi_change = 0
                c_oi_change_pct = 0.0
                c_ltp = 0.0
                c_ltp_change = 0.0
                c_vol = 0
                c_vol_change = 0
                c_buildup = BuildupType.NEUTRAL
                c_bias = DirectionalBias.NEUTRAL

            # Put analysis
            p_curr = s.put
            p_prev = prev_s.put if prev_s else None

            if p_curr is not None:
                p_oi_raw = p_curr.get("oi")
                p_ltp_raw = p_curr.get("ltp")
                p_vol_raw = p_curr.get("volume")

                p_oi = int(p_oi_raw) if p_oi_raw is not None else 0
                p_ltp = float(p_ltp_raw) if p_ltp_raw is not None else 0.0
                p_vol = int(p_vol_raw) if p_vol_raw is not None else 0

                if (
                    p_prev is not None
                    and p_oi_raw is not None
                    and p_prev.get("oi") is not None
                    and p_ltp_raw is not None
                    and p_prev.get("ltp") is not None
                ):
                    p_prev_oi = int(p_prev.get("oi", p_oi))
                    p_prev_ltp = float(p_prev.get("ltp", p_ltp))
                    p_prev_vol = int(p_prev.get("volume", p_vol))

                    p_oi_change = p_oi - p_prev_oi
                    p_oi_change_pct = (p_oi_change / p_prev_oi * 100.0) if p_prev_oi > 0 else 0.0
                    p_ltp_change = p_ltp - p_prev_ltp
                    p_ltp_change_pct = (p_ltp_change / p_prev_ltp * 100.0) if p_prev_ltp > 0 else 0.0
                    p_vol_change = p_vol - p_prev_vol
                else:
                    # Newly appearing strike or missing data: establish baseline, zero delta
                    p_oi_change = 0
                    p_oi_change_pct = 0.0
                    p_ltp_change = 0.0
                    p_ltp_change_pct = 0.0
                    p_vol_change = 0

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
            else:
                p_oi = 0
                p_oi_change = 0
                p_oi_change_pct = 0.0
                p_ltp = 0.0
                p_ltp_change = 0.0
                p_vol = 0
                p_vol_change = 0
                p_buildup = BuildupType.NEUTRAL
                p_bias = DirectionalBias.NEUTRAL

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
            underlying=underlying,
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
            observation_interval_seconds=interval_seconds,
            is_stale=is_stale,
        )
        self._profile_history[key].append(profile)
        self.latest_profiles[key] = profile
        return profile

    def analyze_concentration_near_level(
        self,
        level: float,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None,
        tolerance_points: float = 50.0,
    ) -> Dict[str, Any]:
        """Analyzes Call vs Put OI accumulation and net flow near a specific technical level as of cutoff."""
        profile = self.get_profile(underlying=underlying, expiry=expiry, as_of=as_of)
        if not profile or not profile.strikes or profile.is_stale:
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
            call_oi_change >= self.min_opt_oi_change_contracts
            and any(s.call_buildup == BuildupType.SHORT_BUILDUP for s in matching)
        )
        # Put writers are active if Put OI increased with Put price falling (support writing)
        put_writers_active = (
            put_oi_change >= self.min_opt_oi_change_contracts
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

    def get_profile(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None,
    ) -> Optional[OIProfile]:
        """Returns the OI profile for an underlying and expiry, causal as of a cutoff timestamp."""
        matching_keys = [
            k for k in self._profile_history if k[0].upper() == underlying.upper() and (expiry is None or k[1] == expiry)
        ]
        if not matching_keys:
            return None

        # Exclude expired contracts relative to query date
        query_date = as_of.date() if as_of is not None else (self.current_session_date or date.today())
        active_keys = [
            k for k in matching_keys if k[1] >= query_date
        ]
        if not active_keys:
            return None

        # Select nearest active expiry
        active_keys.sort(key=lambda k: k[1])
        key = active_keys[0]

        history = self._profile_history[key]
        if not history:
            return None

        if as_of is None:
            return history[-1]

        # Reverse lookup for snapshot on or before as_of
        for prof in reversed(history):
            if prof.timestamp <= as_of:
                # Dynamically recompute freshness relative to as_of
                age = (as_of - prof.timestamp).total_seconds()
                is_stale = age > self.max_snapshot_age_seconds
                return replace(prof, is_stale=is_stale)

        return None

    def get_latest_profile(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
    ) -> Optional[OIProfile]:
        """Backward-compatible helper returning latest profile."""
        return self.get_profile(underlying=underlying, expiry=expiry, as_of=None)

    def _classify_buildup(
        self,
        price_change: float,
        price_change_pct: float,
        oi_change: int,
        oi_change_pct: float,
        instrument_type: str,  # "FUT", "CE", "PE"
    ) -> Tuple[BuildupType, DirectionalBias]:
        """Classifies positioning heuristic and assigns strict directional bias per instrument type.

        Crucial rule: For a non-neutral buildup, BOTH price change AND OI change must be
        statistically significant! If price is flat or OI is flat, it is NEUTRAL.
        """
        if instrument_type == "FUT":
            is_oi_significant = (
                abs(oi_change) >= self.min_fut_oi_change_contracts
                and abs(oi_change_pct) >= self.min_fut_oi_change_threshold_pct
            )
            is_price_significant = (
                abs(price_change) >= self.min_fut_price_change_points
                or abs(price_change_pct) >= self.min_fut_price_change_pct
            )
        else:
            is_oi_significant = (
                abs(oi_change) >= self.min_opt_oi_change_contracts
                and abs(oi_change_pct) >= self.min_opt_oi_change_threshold_pct
            )
            is_price_significant = (
                abs(price_change) >= self.min_opt_price_change_points
                or abs(price_change_pct) >= self.min_opt_price_change_pct
            )

        # Both variables must have non-zero, statistically meaningful movement
        if not is_oi_significant or not is_price_significant:
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
        """Calculates optional descriptive Max Pain strike."""
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

                if k > strike_price:
                    total_payout += call_oi * (k - strike_price)
                if k < strike_price:
                    total_payout += put_oi * (strike_price - k)

            if total_payout < min_total_payout:
                min_total_payout = total_payout
                max_pain_strike = k

        return max_pain_strike
