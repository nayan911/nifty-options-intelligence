"""Put-Call Ratio (PCR) Intelligence Engine.

Computes aggregate OI PCR, near-ATM OI PCR, cumulative volume PCR,
interval volume PCR, strike-weighted PCR, and comparable-basket Delta PCR (ΔPCR).
Safeguards against zero denominators, spurious delta artifacts from ATM shifts,
and supports causal, availability-aware retrieval with dynamic freshness checking.
"""

import math
from collections import deque
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Deque, Dict, Optional, Tuple

from nifty_engine.core.interfaces import OptionChainSnapshot


@dataclass
class PCRSnapshot:
    """Consolidated PCR metrics for an option chain snapshot."""

    underlying: str
    timestamp: datetime
    expiry: date
    spot_price: float

    # Core PCR metrics (None if denominator is zero)
    aggregate_oi_pcr: Optional[float]
    near_atm_oi_pcr: Optional[float]
    cumulative_volume_pcr: Optional[float]
    interval_volume_pcr: Optional[float]
    strike_weighted_pcr: Optional[float]

    # Delta PCR (change from previous comparable observation)
    delta_oi_pcr: Optional[float]
    delta_weighted_pcr: Optional[float]

    # Raw aggregates
    total_put_oi: int
    total_call_oi: int
    total_put_volume: int
    total_call_volume: int

    # Metadata
    strikes_evaluated_count: int
    is_stale: bool = False


class PCREngine:
    """Evaluates multi-dimensional Put-Call Ratios with strike weighting

    and invariant-safe Delta PCR calculations.
    """

    def __init__(
        self,
        near_atm_strike_range: int = 5,
        weighting_decay_points: float = 150.0,
        max_snapshot_age_seconds: float = 360.0,
        history_max_len: int = 100,
    ):
        self.near_atm_strike_range = near_atm_strike_range
        self.weighting_decay_points = weighting_decay_points
        self.max_snapshot_age_seconds = max_snapshot_age_seconds
        self.history_max_len = history_max_len

        # Current session date tracking
        self.current_session_date: Optional[date] = None

        # Expiry-isolated history of raw snapshots and computed PCRs: {(underlying, expiry): deque}
        self._snapshot_history: Dict[Tuple[str, date], Deque[OptionChainSnapshot]] = {}
        self._pcr_history: Dict[Tuple[str, date], Deque[PCRSnapshot]] = {}
        self.latest_pcr: Dict[Tuple[str, date], PCRSnapshot] = {}

    def on_option_chain(
        self,
        snapshot: OptionChainSnapshot,
        as_of: Optional[datetime] = None,
    ) -> Optional[PCRSnapshot]:
        """Processes an incoming OptionChainSnapshot and computes all PCR metrics."""
        underlying = snapshot.underlying.upper()
        expiry = snapshot.expiry
        current_time = as_of or snapshot.api_timestamp

        key = (underlying, expiry)
        if key not in self._snapshot_history:
            self._snapshot_history[key] = deque(maxlen=self.history_max_len)
            self._pcr_history[key] = deque(maxlen=self.history_max_len)

        snap_history = self._snapshot_history[key]

        # Session date update
        is_new_session = (
            self.current_session_date is None
            or current_time.date() > self.current_session_date
            or (bool(snap_history) and snap_history[-1].api_timestamp.date() < current_time.date())
        )
        if self.current_session_date is None or current_time.date() > self.current_session_date:
            self.current_session_date = current_time.date()

        # Ignore expired contracts
        if expiry < self.current_session_date:
            return None

        # Reject duplicate or out-of-order snapshots
        if snap_history and snapshot.api_timestamp <= snap_history[-1].api_timestamp:
            return self._pcr_history[key][-1] if self._pcr_history[key] else None

        prev_snapshot: Optional[OptionChainSnapshot] = None
        if snap_history and not is_new_session:
            prev_snapshot = snap_history[-1]
        snap_history.append(snapshot)

        is_stale = (current_time - snapshot.api_timestamp).total_seconds() > self.max_snapshot_age_seconds

        spot = snapshot.spot_price
        strikes = snapshot.strikes

        total_call_oi = sum((s.call.get("oi") or 0) for s in strikes if s.call)
        total_put_oi = sum((s.put.get("oi") or 0) for s in strikes if s.put)
        total_call_vol = sum((s.call.get("volume") or 0) for s in strikes if s.call)
        total_put_vol = sum((s.put.get("volume") or 0) for s in strikes if s.put)

        # 1. Aggregate OI PCR: Sum(Put OI) / Sum(Call OI)
        agg_oi_pcr = round(total_put_oi / total_call_oi, 4) if total_call_oi > 0 else None

        # 2. Cumulative Volume PCR: Sum(Put Vol) / Sum(Call Vol)
        cum_vol_pcr = round(total_put_vol / total_call_vol, 4) if total_call_vol > 0 else None

        # 3. Near-ATM OI PCR: restricted to ATM ± N strikes
        sorted_strikes = sorted(strikes, key=lambda s: s.strike)
        sorted_by_dist = sorted(sorted_strikes, key=lambda s: abs(s.strike - spot))
        near_atm_strikes = sorted_by_dist[: 2 * self.near_atm_strike_range + 1]

        near_atm_call_oi = sum((s.call.get("oi") or 0) for s in near_atm_strikes if s.call)
        near_atm_put_oi = sum((s.put.get("oi") or 0) for s in near_atm_strikes if s.put)
        near_atm_pcr = round(near_atm_put_oi / near_atm_call_oi, 4) if near_atm_call_oi > 0 else None

        # 4. Strike-Weighted PCR: strikes closer to ATM have higher weight
        weighted_put_sum = 0.0
        weighted_call_sum = 0.0
        for s in strikes:
            dist = abs(s.strike - spot)
            w = math.exp(-0.5 * (dist / self.weighting_decay_points) ** 2)
            c_oi = (s.call.get("oi") or 0) if s.call else 0
            p_oi = (s.put.get("oi") or 0) if s.put else 0
            weighted_call_sum += w * c_oi
            weighted_put_sum += w * p_oi

        weighted_pcr = (
            round(weighted_put_sum / weighted_call_sum, 4) if weighted_call_sum > 0.0 else None
        )

        # 5. Interval Volume PCR & Invariant-Safe Delta PCR (ΔPCR)
        interval_vol_pcr = None
        delta_oi_pcr = None
        delta_weighted_pcr = None

        if prev_snapshot is not None:
            # Map previous strikes
            prev_map = {s.strike: s for s in prev_snapshot.strikes}

            # Interval Volume PCR across all matching strikes with valid data in both
            int_put_vol = 0
            int_call_vol = 0
            for s in strikes:
                if s.strike in prev_map and s.call and s.put and prev_map[s.strike].call and prev_map[s.strike].put:
                    prev_s = prev_map[s.strike]
                    c_curr_vol = s.call.get("volume") or 0
                    c_prev_vol = prev_s.call.get("volume") or 0
                    p_curr_vol = s.put.get("volume") or 0
                    p_prev_vol = prev_s.put.get("volume") or 0

                    int_call_vol += max(0, c_curr_vol - c_prev_vol)
                    int_put_vol += max(0, p_curr_vol - p_prev_vol)

            if int_call_vol > 0:
                interval_vol_pcr = round(int_put_vol / int_call_vol, 4)

            # Invariant-Safe ΔPCR:
            # Must be evaluated on the COMPARABLE STRIKE BASKET (intersection of strikes where both CE and PE existed with non-None OI)
            common_strikes = [
                s for s in strikes
                if s.strike in prev_map
                and s.call and s.put
                and prev_map[s.strike].call and prev_map[s.strike].put
                and s.call.get("oi") is not None and s.put.get("oi") is not None
                and prev_map[s.strike].call.get("oi") is not None and prev_map[s.strike].put.get("oi") is not None
            ]
            if common_strikes:
                curr_common_call_oi = sum(s.call.get("oi") or 0 for s in common_strikes)
                curr_common_put_oi = sum(s.put.get("oi") or 0 for s in common_strikes)

                prev_common_call_oi = sum(prev_map[s.strike].call.get("oi") or 0 for s in common_strikes)
                prev_common_put_oi = sum(prev_map[s.strike].put.get("oi") or 0 for s in common_strikes)

                if curr_common_call_oi > 0 and prev_common_call_oi > 0:
                    curr_common_pcr = curr_common_put_oi / curr_common_call_oi
                    prev_common_pcr = prev_common_put_oi / prev_common_call_oi
                    delta_oi_pcr = round(curr_common_pcr - prev_common_pcr, 4)

                # Consistent weighted delta
                w_curr_put = 0.0
                w_curr_call = 0.0
                w_prev_put = 0.0
                w_prev_call = 0.0

                for s in common_strikes:
                    dist = abs(s.strike - spot)
                    w = math.exp(-0.5 * (dist / self.weighting_decay_points) ** 2)

                    c_c = s.call.get("oi", 0)
                    p_c = s.put.get("oi", 0)
                    c_p = prev_map[s.strike].call.get("oi", 0)
                    p_p = prev_map[s.strike].put.get("oi", 0)

                    w_curr_call += w * c_c
                    w_curr_put += w * p_c
                    w_prev_call += w * c_p
                    w_prev_put += w * p_p

                if w_curr_call > 0.0 and w_prev_call > 0.0:
                    curr_w_pcr = w_curr_put / w_curr_call
                    prev_w_pcr = w_prev_put / w_prev_call
                    delta_weighted_pcr = round(curr_w_pcr - prev_w_pcr, 4)

        pcr_snap = PCRSnapshot(
            underlying=underlying,
            timestamp=snapshot.api_timestamp,
            expiry=expiry,
            spot_price=spot,
            aggregate_oi_pcr=agg_oi_pcr,
            near_atm_oi_pcr=near_atm_pcr,
            cumulative_volume_pcr=cum_vol_pcr,
            interval_volume_pcr=interval_vol_pcr,
            strike_weighted_pcr=weighted_pcr,
            delta_oi_pcr=delta_oi_pcr,
            delta_weighted_pcr=delta_weighted_pcr,
            total_put_oi=total_put_oi,
            total_call_oi=total_call_oi,
            total_put_volume=total_put_vol,
            total_call_volume=total_call_vol,
            strikes_evaluated_count=len(strikes),
            is_stale=is_stale,
        )

        self._pcr_history[key].append(pcr_snap)
        self.latest_pcr[key] = pcr_snap
        return pcr_snap

    def get_pcr(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
        as_of: Optional[datetime] = None,
    ) -> Optional[PCRSnapshot]:
        """Returns the PCR snapshot, causal as of a cutoff timestamp."""
        matching_keys = [
            k for k in self._pcr_history if k[0].upper() == underlying.upper() and (expiry is None or k[1] == expiry)
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

        history = self._pcr_history[key]
        if not history:
            return None

        if as_of is None:
            return history[-1]

        # Reverse lookup for snapshot on or before as_of
        for snap in reversed(history):
            if snap.timestamp <= as_of:
                # Dynamically recompute freshness relative to as_of
                age = (as_of - snap.timestamp).total_seconds()
                is_stale = age > self.max_snapshot_age_seconds
                return replace(snap, is_stale=is_stale)

        return None

    def get_latest_pcr(
        self,
        underlying: str = "NIFTY",
        expiry: Optional[date] = None,
    ) -> Optional[PCRSnapshot]:
        """Backward-compatible helper returning latest PCR."""
        return self.get_pcr(underlying=underlying, expiry=expiry, as_of=None)
