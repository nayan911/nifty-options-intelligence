"""Derivatives Flow Assessment Engine.

Synthesizes Open Interest flow, futures positioning, PCR dynamics,
and strike concentration into a structured DerivativesAssessment.
Provides diagnostic classifications (SUPPORTIVE, CONFLICTING, NEUTRAL, etc.)
without hard vetoes or alteration of technical confluence scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from nifty_engine.analysis.derivatives.oi_analysis import (
    DirectionalBias,
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
    confidence_score: float  # Descriptive 0.0 - 1.0 confidence in the derivatives flow
    futures_bias: DirectionalBias
    options_bias: DirectionalBias
    pcr_snapshot: Optional[PCRSnapshot]
    oi_profile: Optional[OIProfile]
    reasons: List[str] = field(default_factory=list)
    contra_reasons: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    decision_at: Optional[datetime] = None
    confirmation_status: str = "CONFIRMED"
    is_stale: bool = False

    def to_dict(self) -> dict:
        """Serializes assessment for logging and telemetry."""
        return {
            "assessment_type": self.assessment_type.value,
            "direction": self.direction.value,
            "confidence_score": round(self.confidence_score, 2),
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
    ):
        self.oi_analyzer = oi_analyzer or OpenInterestAnalyzer()
        self.pcr_engine = pcr_engine or PCREngine()

    def evaluate_setup(
        self,
        direction: BreakoutDirection,
        level_price: float,
        as_of: datetime,
        confirmation_status: str = "CONFIRMED",
    ) -> DerivativesAssessment:
        """Evaluates derivatives alignment for a given breakout direction and reference level.

        Strictly descriptive: Produces an assessment classification without imposing hard vetoes.
        """
        oi_profile = self.oi_analyzer.get_latest_profile()
        pcr_snap = self.pcr_engine.get_latest_pcr()

        # Check data sufficiency and freshness
        if oi_profile is None or pcr_snap is None:
            return DerivativesAssessment(
                assessment_type=DerivativesAssessmentType.INSUFFICIENT_DATA,
                direction=direction,
                confidence_score=0.0,
                futures_bias=DirectionalBias.NEUTRAL,
                options_bias=DirectionalBias.NEUTRAL,
                pcr_snapshot=pcr_snap,
                oi_profile=oi_profile,
                reasons=[],
                contra_reasons=["No derivatives option chain or PCR snapshot recorded"],
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
                confidence_score=0.0,
                futures_bias=DirectionalBias.NEUTRAL,
                options_bias=DirectionalBias.NEUTRAL,
                pcr_snapshot=pcr_snap,
                oi_profile=oi_profile,
                reasons=[],
                contra_reasons=["Derivatives option chain snapshot is stale"],
                timestamp=as_of,
                decision_at=as_of,
                confirmation_status=confirmation_status,
                is_stale=True,
            )

        supporting_reasons: List[str] = []
        contra_reasons: List[str] = []

        # 1. Futures Positioning
        fut_bias = DirectionalBias.NEUTRAL
        if oi_profile.futures_analysis is not None:
            fut_bias = oi_profile.futures_analysis.directional_bias
            if direction == BreakoutDirection.BULLISH:
                if fut_bias == DirectionalBias.BULLISH:
                    supporting_reasons.append(
                        f"Futures positioning is Bullish ({oi_profile.futures_analysis.buildup.value})"
                    )
                elif fut_bias == DirectionalBias.BEARISH:
                    contra_reasons.append(
                        f"Futures positioning is Bearish ({oi_profile.futures_analysis.buildup.value})"
                    )
            elif direction == BreakoutDirection.BEARISH:
                if fut_bias == DirectionalBias.BEARISH:
                    supporting_reasons.append(
                        f"Futures positioning is Bearish ({oi_profile.futures_analysis.buildup.value})"
                    )
                elif fut_bias == DirectionalBias.BULLISH:
                    contra_reasons.append(
                        f"Futures positioning is Bullish ({oi_profile.futures_analysis.buildup.value})"
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
        level_info = self.oi_analyzer.analyze_concentration_near_level(level_price)
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

        # 4. Delta PCR Dynamics (ΔPCR)
        if pcr_snap.delta_oi_pcr is not None:
            if direction == BreakoutDirection.BULLISH:
                if pcr_snap.delta_oi_pcr > 0.02:
                    supporting_reasons.append(f"PCR increasing (ΔPCR = +{pcr_snap.delta_oi_pcr:.3f})")
                elif pcr_snap.delta_oi_pcr < -0.02:
                    contra_reasons.append(f"PCR decreasing (ΔPCR = {pcr_snap.delta_oi_pcr:.3f})")
            elif direction == BreakoutDirection.BEARISH:
                if pcr_snap.delta_oi_pcr < -0.02:
                    supporting_reasons.append(f"PCR decreasing (ΔPCR = {pcr_snap.delta_oi_pcr:.3f})")
                elif pcr_snap.delta_oi_pcr > 0.02:
                    contra_reasons.append(f"PCR increasing (ΔPCR = +{pcr_snap.delta_oi_pcr:.3f})")

        # 5. Interval Volume PCR
        if pcr_snap.interval_volume_pcr is not None:
            if direction == BreakoutDirection.BULLISH and pcr_snap.interval_volume_pcr > 1.1:
                supporting_reasons.append(
                    f"Interval volume PCR is elevated ({pcr_snap.interval_volume_pcr:.2f})"
                )
            elif direction == BreakoutDirection.BEARISH and pcr_snap.interval_volume_pcr < 0.9:
                supporting_reasons.append(
                    f"Interval volume PCR is depressed ({pcr_snap.interval_volume_pcr:.2f})"
                )

        # Classify overall assessment
        num_support = len(supporting_reasons)
        num_contra = len(contra_reasons)

        if num_support >= 3 and num_contra == 0:
            assessment_type = DerivativesAssessmentType.STRONGLY_SUPPORTIVE
            confidence = 0.9
        elif num_support > num_contra and num_support >= 1:
            assessment_type = DerivativesAssessmentType.SUPPORTIVE
            confidence = 0.7
        elif num_contra >= 3 and num_support == 0:
            assessment_type = DerivativesAssessmentType.STRONGLY_CONFLICTING
            confidence = 0.1
        elif num_contra > num_support and num_contra >= 1:
            assessment_type = DerivativesAssessmentType.CONFLICTING
            confidence = 0.3
        else:
            assessment_type = DerivativesAssessmentType.NEUTRAL
            confidence = 0.5

        return DerivativesAssessment(
            assessment_type=assessment_type,
            direction=direction,
            confidence_score=confidence,
            futures_bias=fut_bias,
            options_bias=options_bias,
            pcr_snapshot=pcr_snap,
            oi_profile=oi_profile,
            reasons=supporting_reasons,
            contra_reasons=contra_reasons,
            timestamp=pcr_snap.timestamp,
            decision_at=as_of,
            confirmation_status=confirmation_status,
            is_stale=False,
        )
