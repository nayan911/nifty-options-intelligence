"""S000 strategy parameters: one validated, immutable object the technical engine is built from.

Every default below equals the value that was hard-coded in the engine before the strategy
registry existed (commit 183ef0a), so `S000Params()` reproduces S000 exactly. The registry file
(strategies/registry.yaml) states the same values explicitly; the loader validates them strictly
(unknown keys, wrong types, out-of-range or contradictory values are rejected).

Two kinds of settings:
- Tunable params (the models below): wired into the engine; changing one changes behaviour, and
  therefore requires a new strategy version in the registry.
- FIXED definitions (S000_FIXED): indicator definitions the engine computes internally (EMA/RSI/
  ATR/RVOL periods, history depth, pairing tolerance). They are declared in the registry for
  transparency and must match the engine exactly; changing them needs a code change and a new
  strategy, not a YAML edit.
"""

from typing import Any, Dict, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LevelParams(_Strict):
    """Support/resistance zones: half-width = max(zone_atr_multiple * ATR, min_zone_width_points)."""
    zone_atr_multiple: float = Field(0.15, gt=0, le=2)
    min_zone_width_points: float = Field(3.0, ge=0, le=100)


class SwingParams(_Strict):
    """Pivot swing confirmation: bars required on each side of a pivot."""
    left_bars: int = Field(ge=1, le=20)
    right_bars: int = Field(ge=1, le=20)


class SwingSet(_Strict):
    tf_1m: SwingParams = Field(SwingParams(left_bars=3, right_bars=3), alias="1m")
    tf_5m: SwingParams = Field(SwingParams(left_bars=3, right_bars=3), alias="5m")
    tf_15m: SwingParams = Field(SwingParams(left_bars=2, right_bars=2), alias="15m")


class BreakoutParams(_Strict):
    """Breakout -> retest state machine on one timeframe."""
    min_breakout_rvol: float = Field(1.3, gt=0, le=10)          # futures RVOL on the breakout bar
    min_body_atr_ratio: float = Field(0.35, ge=0, le=5)          # breakout body >= ratio * ATR
    approach_buffer_atr: float = Field(0.25, ge=0, le=5)         # retest "near level" distance (ATR)
    min_approach_points: float = Field(4.0, ge=0, le=100)        # ... but at least this many points
    max_retest_candles: int = Field(ge=1, le=100)                # retest must happen within N bars
    stop_buffer_atr: float = Field(0.1, ge=0, le=5)              # setup stop beyond retest bar extreme
    min_setup_risk_points: float = Field(10.0, ge=0, le=500)     # setup target uses risk >= this
    setup_target_rr: float = Field(1.5, gt=0, le=20)             # setup's own target (R multiple)


class BreakoutSet(_Strict):
    tf_1m: BreakoutParams = Field(BreakoutParams(max_retest_candles=6), alias="1m")
    tf_5m: BreakoutParams = Field(BreakoutParams(max_retest_candles=4), alias="5m")


class QualificationParams(_Strict):
    """5m setup qualification against the 15m regime."""
    min_headroom_atr_multiple: float = Field(1.5, ge=0, le=20)   # room to the opposing 15m level ...
    min_headroom_points: float = Field(25.0, ge=0, le=1000)      # ... at least max(mult * 5m ATR, points)
    max_15m_macro_age_seconds: float = Field(1200.0, gt=0, le=7200)
    max_5m_futures_age_seconds: float = Field(300.0, gt=0, le=3600)


class EntryParams(_Strict):
    """1m entry trigger."""
    entry_window_minutes: int = Field(15, ge=1, le=120)
    max_1m_futures_age_seconds: float = Field(60.0, gt=0, le=600)
    fallback_stop_points: float = Field(25.0, gt=0, le=500)      # used only if a setup carries no stop
    rsi_overbought: float = Field(75.0, gt=50, le=100)           # no bullish entry above this 1m RSI
    rsi_oversold: float = Field(25.0, ge=0, lt=50)               # no bearish entry below this 1m RSI
    min_risk_points: float = Field(8.0, gt=0, le=500)            # stop at least this far from entry


class TargetParams(_Strict):
    target_1_rr: float = Field(1.5, gt=0, le=20)  # also: opposing 15m level must not block target 1
    target_2_rr: float = Field(2.0, gt=0, le=20)  # capped at the opposing 15m level

    @model_validator(mode="after")
    def _ordered(self):
        if self.target_2_rr < self.target_1_rr:
            raise ValueError("target_2_rr must be >= target_1_rr")
        return self


class ScoringParams(_Strict):
    """0-100 confluence score; missing data scores 0. A candidate needs score >= min_score."""
    min_score: float = Field(60.0, ge=0, le=100)
    trend_aligned: float = Field(30.0, ge=0, le=100)
    trend_aligned_provisional: float = Field(20.0, ge=0, le=100)
    trend_sideways: float = Field(15.0, ge=0, le=100)
    # (minimum RVOL, points), highest tier first; the first tier met scores.
    rvol_tiers: Tuple[Tuple[float, float], ...] = ((2.0, 25.0), (1.5, 20.0), (1.3, 15.0))
    vwap_5m: float = Field(10.0, ge=0, le=100)
    vwap_1m: float = Field(10.0, ge=0, le=100)
    # (minimum headroom in 5m ATRs, points), highest tier first.
    headroom_tiers: Tuple[Tuple[float, float], ...] = ((2.0, 15.0), (1.5, 10.0))
    rsi_bull_range: Tuple[float, float] = (50.0, 68.0)
    rsi_bear_range: Tuple[float, float] = (32.0, 50.0)
    rsi_points: float = Field(10.0, ge=0, le=100)

    @model_validator(mode="after")
    def _consistent(self):
        for name in ("rvol_tiers", "headroom_tiers"):
            tiers = getattr(self, name)
            if not tiers:
                raise ValueError(f"{name} must not be empty")
            if any(t[0] < 0 or t[1] < 0 for t in tiers):
                raise ValueError(f"{name} values must be >= 0")
            if list(tiers) != sorted(tiers, key=lambda t: -t[0]) or len({t[0] for t in tiers}) != len(tiers):
                raise ValueError(f"{name} thresholds must be strictly descending")
        for name in ("rsi_bull_range", "rsi_bear_range"):
            lo, hi = getattr(self, name)
            if not 0 <= lo <= hi <= 100:
                raise ValueError(f"{name} must satisfy 0 <= low <= high <= 100")
        if self.trend_aligned_provisional > self.trend_aligned:
            raise ValueError("trend_aligned_provisional must be <= trend_aligned")
        if self.max_score() < self.min_score:
            raise ValueError(f"min_score {self.min_score} is unreachable (max possible {self.max_score()})")
        return self

    def max_score(self) -> float:
        best = (self.trend_aligned + max(p for _, p in self.rvol_tiers) + self.vwap_5m + self.vwap_1m
                + max(p for _, p in self.headroom_tiers) + self.rsi_points)
        return min(100.0, best)


class S000Params(_Strict):
    levels: LevelParams = LevelParams()
    swings: SwingSet = SwingSet()
    breakouts: BreakoutSet = BreakoutSet()
    qualification: QualificationParams = QualificationParams()
    entry: EntryParams = EntryParams()
    targets: TargetParams = TargetParams()
    scoring: ScoringParams = ScoringParams()

    @model_validator(mode="after")
    def _cross(self):
        if self.entry.min_risk_points >= self.entry.fallback_stop_points:
            raise ValueError("entry.min_risk_points must be < entry.fallback_stop_points")
        return self

    def canonical(self) -> Dict[str, Any]:
        """JSON-ready dict with YAML key names (used for hashing and provenance)."""
        return self.model_dump(by_alias=True, mode="json")


# Indicator definitions computed inside the engine (not tunable from YAML; see module docstring).
S000_FIXED: Dict[str, Any] = {
    "ema_periods": [9, 21, 50, 200],
    "rsi_period": 14,
    "atr_period": 14,
    "rvol_lookback_bars": 20,
    "warmup_min_bars": 21,
    "history_bars": {"1m": 400, "5m": 200, "15m": 100},
    "spot_futures_pairing_tolerance_seconds": 5.0,
    "trend_bias": "EMA9 vs EMA21 with close vs EMA9; provisional structural fallback (opening range, "
                  "daily references, swings) during warmup",
}
