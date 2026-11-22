"""Derivatives Flow Assessment Engine.

Synthesizes Open Interest flow, futures positioning, PCR dynamics,
and strike concentration into a structured DerivativesAssessment.
Provides diagnostic classifications (SUPPORTIVE, CONFLICTING, NEUTRAL, etc.)
with dynamic freshness validation and zero hard vetoes on technical confluence scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from nifty_engine.analysis.derivatives.oi_analysis import (
    DirectionalBias,
    FuturesOIAnalysis,
    OIProfile,
    OpenInterestAnalyzer,
)
from nifty_engine.analysis.derivatives.pcr import PCREngine, PCRSnapshot
from nifty_engine.analysis.technical.breakouts import BreakoutDirection


class DerivativesAssessmentType(str, Enum):
    """Categorical alignment between derivatives positioning and technical setup."""

    STRONGLY_SUPPORTIVE = "STRONGLY_SUPPORTIVE"
    SUPPORTIVE = "SUPPORTIVE"
    NEUTRAL = "NEUTRAL"
    CONFLICTING = "CONFLICTING"
    STRONGLY_CONFLICTING = "STRONGLY_CONFLICTING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class DerivativesAssessment:
    """Structured derivatives intelligence evaluation for a technical setup."""

    assessment_type: DerivativesAssessmentType
    direction: BreakoutDirection
    alignment_score: float  # Descriptive 0.0 - 1.0 degree of directional alignment
    data_quality_score: float  # 0.0 - 1.0 measure of data completeness and freshness
    futures_bias: DirectionalBias
    options_bias: DirectionalBias
    pcr_snapshot: Optional[PCRSnapshot]
    oi_profile: Optional[OIProfile]
    futures_analysis: Optional[FuturesOIAnalysis] = None
    reasons: List[str] = field(default_factory=list)
    contra_reasons: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    decision_at: Optional[datetime] = None
    confirmation_status: str = "CONFIRMED"
    is_stale: bool = False

    @property
    def confidence_score(self) -> float:
        """Backward-compatible property alias for alignment_score."""
        return self.alignment_score

    def to_dict(self) -> dict:
        """Serializes assessment for logging and telemetry."""
        return {
            "assessment_type": self.assessment_type.value,
            "direction": self.direction.value,
            "alignment_score": round(self.alignment_score, 2),
            "data_quality_score": round(self.data_quality_score, 2),
            "futures_bias": self.futures_bias.value,
            "options_bias": self.options_bias.value,
            "reasons": self.reasons,
            "contra_reasons": self.contra_reasons,
            "confirmation_status": self.confirmation_status,
            "is_stale": self.is_stale,
            "aggregate_oi_pcr": self.pcr_snapshot.aggregate_oi_pcr if self.pcr_snapshot else None,
            "delta_oi_pcr": self.pcr_snapshot.delta_oi_pcr if self.pcr_snapshot else None,
            "call_wall": self.oi_profile.call_wall_strike if self.oi_profile else None,
            "put_wall": self.oi_profile.put_wall_strike if self.oi_profile else None,
        }


class DerivativesIntelligenceEngine:
    """Coordinates OI analysis and PCR calculations to generate DerivativesAssessments."""

    def __init__(
        self,
        oi_analyzer: Optional[OpenInterestAnalyzer] = None,
        pcr_engine: Optional[PCREngine] = None,
        primary_futures: str = "NSE:NIFTY26NOVFUT",
    ):
        self.oi_analyzer = oi_analyzer or OpenInterestAnalyzer()
        self.pcr_engine = pcr_engine or PCREngine()
        self.primary_futures = primary_futures

    def set_futures_ticker(self, futures_ticker: str) -> None:
        """Updates active primary futures symbol (e.g. upon monthly expiry rollover)."""
        self.primary_futures = futures_ticker

    def evaluate_setup(
        self,
        direction: BreakoutDirection,
        level_price: float,
        as_of: datetime,
        underlying: str = "NIFTY",
        futures_symbol: Optional[str] = None,
        confirmation_status: str = "CONFIRMED",
    ) -> DerivativesAssessment:
        """Evaluates derivatives alignment for a given breakout direction and reference level.

        Strictly descriptive: Causal as of cutoff `as_of`, with dynamic freshness validation
        and zero hard vetoes.
        """
        active_futures = futures_symbol or self.primary_futures
        # Causal retrieval as of as_of
        oi_profile = self.oi_analyzer.get_profile(underlying=underlying, as_of=as_of)
        pcr_snap = self.pcr_engine.get_pcr(underlying=underlying, as_of=as_of)
        fut_analysis = self.oi_analyzer.get_futures_analysis(
            symbol=active_futures, timeframe="5m", as_of=as_of
        )
        if fut_analysis is None:
            # Fallback to 1m futures analysis if 5m is not ready
            fut_analysis = self.oi_analyzer.get_futures_analysis(
                symbol=active_futures, timeframe="1m", as_of=as_of
            )

        # Check data sufficiency and dynamic freshness relative to as_of
        if oi_profile is None or pcr_snap is None:
            return DerivativesAssessment(
                assessment_type=DerivativesAssessmentType.INSUFFICIENT_DATA,
                direction=direction,
                alignment_score=0.5,
                data_quality_score=0.0,
                futures_bias=fut_analysis.directional_bias if fut_analysis else DirectionalBias.NEUTRAL,
                options_bias=DirectionalBias.NEUTRAL,
                pcr_snapshot=pcr_snap,
                oi_profile=oi_profile,
                futures_analysis=fut_analysis,
                reasons=[],
                contra_reasons=["No derivatives option chain or PCR snapshot recorded as of decision time"],
                timestamp=as_of,
                decision_at=as_of,
                confirmation_status=confirmation_status,
                is_stale=True,
            )

        is_stale = oi_profile.is_stale or pcr_snap.is_stale
        if is_stale:
            return DerivativesAssessment(
                assessment_type=DerivativesAssessmentType.INSUFFICIENT_DATA,
                direction=direction,
                alignment_score=0.5,
                data_quality_score=0.2,
                futures_bias=fut_analysis.directional_bias if fut_analysis else DirectionalBias.NEUTRAL,
                options_bias=DirectionalBias.NEUTRAL,
                pcr_snapshot=pcr_snap,
                oi_profile=oi_profile,
                futures_analysis=fut_analysis,
                reasons=[],
                contra_reasons=["Derivatives option chain snapshot is stale relative to decision time"],
                timestamp=as_of,
                decision_at=as_of,
                confirmation_status=confirmation_status,
                is_stale=True,
            )

        # Compute dynamic data quality score (0.0 to 1.0) based on valid non-None observations
        age_seconds = max(0.0, (as_of - oi_profile.timestamp).total_seconds())
        freshness_factor = max(0.0, 1.0 - (age_seconds / self.oi_analyzer.max_snapshot_age_seconds))
        expected_strikes = max(1, 2 * self.oi_analyzer.near_atm_strikes_count + 1)
        valid_strikes_count = sum(
            1 for s in oi_profile.near_atm_strikes
            if s.call_oi > 0 and s.put_oi > 0 and s.call_ltp > 0.0 and s.put_ltp > 0.0
        )
        strike_coverage = min(1.0, valid_strikes_count / expected_strikes)
        fut_present = 1.0 if fut_analysis is not None else 0.5
        data_quality = round(0.4 * freshness_factor + 0.3 * strike_coverage + 0.3 * fut_present, 2)

        supporting_reasons: List[str] = []
        contra_reasons: List[str] = []

        # 1. Futures Positioning (independent observation as of as_of)
        fut_bias = DirectionalBias.NEUTRAL
        if fut_analysis is not None:
            fut_bias = fut_analysis.directional_bias
            if direction == BreakoutDirection.BULLISH:
                if fut_bias == DirectionalBias.BULLISH:
                    supporting_reasons.append(
                        f"Futures positioning is Bullish ({fut_analysis.buildup.value})"
                    )
                elif fut_bias == DirectionalBias.BEARISH:
                    contra_reasons.append(
                        f"Futures positioning is Bearish ({fut_analysis.buildup.value})"
                    )
            elif direction == BreakoutDirection.BEARISH:
                if fut_bias == DirectionalBias.BEARISH:
                    supporting_reasons.append(
                        f"Futures positioning is Bearish ({fut_analysis.buildup.value})"
                    )
                elif fut_bias == DirectionalBias.BULLISH:
                    contra_reasons.append(
                        f"Futures positioning is Bullish ({fut_analysis.buildup.value})"
                    )

        # 2. Options Near-ATM Aggregate Bias
        options_bias = oi_profile.aggregate_bias
        if direction == BreakoutDirection.BULLISH:
            if options_bias == DirectionalBias.BULLISH:
                supporting_reasons.append("Near-ATM option strikes show net Bullish positioning")
            elif options_bias == DirectionalBias.BEARISH:
                contra_reasons.append("Near-ATM option strikes show net Bearish positioning")
        elif direction == BreakoutDirection.BEARISH:
            if options_bias == DirectionalBias.BEARISH:
                supporting_reasons.append("Near-ATM option strikes show net Bearish positioning")
            elif options_bias == DirectionalBias.BULLISH:
                contra_reasons.append("Near-ATM option strikes show net Bullish positioning")

        # 3. Level Concentration Analysis (Breakout resistance/support defense)
        level_info = self.oi_analyzer.analyze_concentration_near_level(
            level=level_price, underlying=underlying, expiry=oi_profile.expiry, as_of=as_of
        )
        if level_info.get("is_available"):
            if direction == BreakoutDirection.BULLISH:
                if level_info.get("call_writers_active"):
                    contra_reasons.append(
                        f"Call writers actively building at breakout level {level_price} (+{level_info.get('call_oi_change')} OI)"
                    )
                if level_info.get("put_writers_active"):
                    supporting_reasons.append(
                        f"Put writers supporting breakout level {level_price} (+{level_info.get('put_oi_change')} OI)"
                    )
            elif direction == BreakoutDirection.BEARISH:
                if level_info.get("put_writers_active"):
                    contra_reasons.append(
                        f"Put writers actively defending breakdown level {level_price} (+{level_info.get('put_oi_change')} OI)"
                    )
                if level_info.get("call_writers_active"):
                    supporting_reasons.append(
                        f"Call writers adding pressure at breakdown level {level_price} (+{level_info.get('call_oi_change')} OI)"
                    )

        # 4. Delta PCR Dynamics (ΔPCR on comparable strike basket)
        if pcr_snap.delta_oi_pcr is not None:
            if direction == BreakoutDirection.BULLISH:
                if pcr_snap.delta_oi_pcr > 0.02:
                    supporting_reasons.append(f"PCR increasing on comparable basket (ΔPCR = +{pcr_snap.delta_oi_pcr:.3f})")
                elif pcr_snap.delta_oi_pcr < -0.02:
                    contra_reasons.append(f"PCR decreasing on comparable basket (ΔPCR = {pcr_snap.delta_oi_pcr:.3f})")
            elif direction == BreakoutDirection.BEARISH:
                if pcr_snap.delta_oi_pcr < -0.02:
                    supporting_reasons.append(f"PCR decreasing on comparable basket (ΔPCR = {pcr_snap.delta_oi_pcr:.3f})")
                elif pcr_snap.delta_oi_pcr > 0.02:
                    contra_reasons.append(f"PCR increasing on comparable basket (ΔPCR = +{pcr_snap.delta_oi_pcr:.3f})")

        # Classify overall assessment
        num_support = len(supporting_reasons)
        num_contra = len(contra_reasons)

        if num_support >= 3 and num_contra == 0:
            assessment_type = DerivativesAssessmentType.STRONGLY_SUPPORTIVE
            alignment = 0.9
        elif num_support > num_contra and num_support >= 1:
            assessment_type = DerivativesAssessmentType.SUPPORTIVE
            alignment = 0.7
        elif num_contra >= 3 and num_support == 0:
            assessment_type = DerivativesAssessmentType.STRONGLY_CONFLICTING
            alignment = 0.1
        elif num_contra > num_support and num_contra >= 1:
            assessment_type = DerivativesAssessmentType.CONFLICTING
            alignment = 0.3
        else:
            assessment_type = DerivativesAssessmentType.NEUTRAL
            alignment = 0.5

        return DerivativesAssessment(
            assessment_type=assessment_type,
            direction=direction,
            alignment_score=alignment,
            data_quality_score=data_quality,
            futures_bias=fut_bias,
            options_bias=options_bias,
            pcr_snapshot=pcr_snap,
            oi_profile=oi_profile,
            futures_analysis=fut_analysis,
            reasons=supporting_reasons,
            contra_reasons=contra_reasons,
            timestamp=pcr_snap.timestamp,
            decision_at=as_of,
            confirmation_status=confirmation_status,
            is_stale=False,
        )
