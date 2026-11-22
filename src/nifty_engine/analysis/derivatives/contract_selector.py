"""Quantitative Option Contract Selector for NIFTY Breakout & Retest Setups.

Selects the single optimal option contract (strike, expiry, CE/PE) based on:
1. Target Delta range (0.35 to 0.55) balancing momentum sensitivity and capital at risk.
2. Order-book liquidity and bid-ask spread friction.
3. Intraday Theta decay budgeting over anticipated holding duration.
4. Premium affordability and leverage sweet-spot.
5. All-in execution cost viability (statutory fees, slippage, and net R:R).
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytz

from nifty_engine.analysis.derivatives.cost_model import (
    ExecutionCosts,
    PremiumCostModel,
    PremiumEstimate,
)
from nifty_engine.analysis.derivatives.greeks import GreeksEngine, OptionGreeks
from nifty_engine.analysis.derivatives.oi_analysis import OIProfile
from nifty_engine.analysis.technical.breakouts import BreakoutDirection

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class SelectedContract:
    """Detailed specification of the selected optimal option contract."""

    symbol: str  # Contract ticker (e.g. "NSE:NIFTY26NOV25100CE")
    underlying: str  # "NIFTY", "BANKNIFTY"
    strike: float  # Selected strike price
    option_type: str  # "CE" or "PE"
    expiry: date  # Contract expiry date
    ltp: float  # Current Last Traded Price
    bid: Optional[float]
    ask: Optional[float]
    spread: float  # Bid-ask spread points
    spread_pct: float  # Spread as percentage of LTP
    volume: int  # Traded volume
    oi: int  # Open interest
    greeks: OptionGreeks  # Option Greeks (Delta, Gamma, Theta, Vega, IV)
    selection_score: float  # Score from 0.0 to 100.0
    selection_reasons: List[str]  # Quantitative rationale for selection
    premium_trajectory: PremiumEstimate  # Taylor-expansion premium targets
    execution_costs: ExecutionCosts  # Indian statutory fees and net R:R
    holding_minutes: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes selected contract for telemetry and paper-trading logging."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "strike": self.strike,
            "option_type": self.option_type,
            "expiry": self.expiry.isoformat(),
            "ltp": self.ltp,
            "bid": self.bid,
            "ask": self.ask,
            "spread": round(self.spread, 2),
            "spread_pct": round(self.spread_pct * 100.0, 2),
            "volume": self.volume,
            "oi": self.oi,
            "delta": self.greeks.delta,
            "gamma": self.greeks.gamma,
            "theta": self.greeks.theta,
            "vega": self.greeks.vega,
            "iv_pct": round(self.greeks.iv * 100.0, 2),
            "selection_score": round(self.selection_score, 1),
            "selection_reasons": self.selection_reasons,
            "target_premium_t1": self.premium_trajectory.target_premium_t1,
            "target_premium_t2": self.premium_trajectory.target_premium_t2,
            "stop_loss_premium": self.premium_trajectory.stop_loss_premium,
            "gross_rr_t1": self.execution_costs.gross_rr_t1,
            "net_rr_t1": self.execution_costs.net_rr_t1,
            "total_friction_rupees": self.execution_costs.total_friction_rupees,
            "is_cost_viable": self.execution_costs.is_cost_viable,
        }


class OptionContractSelector:
    """Evaluates option-chain strikes and selects the highest-scoring liquid contract."""

    def __init__(
        self,
        greeks_engine: Optional[GreeksEngine] = None,
        cost_model: Optional[PremiumCostModel] = None,
        near_atm_range: int = 5,
        target_delta_min: float = 0.35,
        target_delta_max: float = 0.55,
        max_acceptable_spread_pct: float = 0.03,  # Max 3% bid-ask spread
        min_liquidity_volume: int = 2000,
        default_holding_minutes: float = 30.0,
        min_selection_score: float = 50.0,
    ):
        self.greeks_engine = greeks_engine or GreeksEngine()
        self.cost_model = cost_model or PremiumCostModel()
        self.near_atm_range = near_atm_range
        self.target_delta_min = target_delta_min
        self.target_delta_max = target_delta_max
        self.max_acceptable_spread_pct = max_acceptable_spread_pct
        self.min_liquidity_volume = min_liquidity_volume
        self.default_holding_minutes = default_holding_minutes
        self.min_selection_score = min_selection_score

    def select_contract(
        self,
        direction: BreakoutDirection,
        spot_price: float,
        chain_profile: OIProfile,
        futures_price: Optional[float] = None,
        as_of: Optional[datetime] = None,
        risk_points: float = 20.0,
        target_points_t1: float = 30.0,
        target_points_t2: float = 50.0,
        holding_minutes: Optional[float] = None,
    ) -> Optional[SelectedContract]:
        """Selects the single best option contract matching the technical breakout candidate."""
        if not chain_profile or not chain_profile.strikes:
            return None

        current_time = as_of or chain_profile.timestamp
        fut_price = futures_price or spot_price
        opt_type = "CE" if direction == BreakoutDirection.BULLISH else "PE"
        hold_mins = holding_minutes or self.default_holding_minutes

        # Calculate spot index Target and Stop Loss levels
        if direction == BreakoutDirection.BULLISH:
            spot_t1 = spot_price + target_points_t1
            spot_t2 = spot_price + target_points_t2
            spot_sl = spot_price - risk_points
        else:
            spot_t1 = spot_price - target_points_t1
            spot_t2 = spot_price - target_points_t2
            spot_sl = spot_price + risk_points

        # Filter near-ATM strikes
        sorted_strikes = sorted(chain_profile.strikes.keys())
        sorted_by_dist = sorted(sorted_strikes, key=lambda k: abs(k - spot_price))
        candidate_strikes = sorted(sorted_by_dist[: 2 * self.near_atm_range + 1])

        evaluated_contracts: List[SelectedContract] = []

        for strike in candidate_strikes:
            strike_data = chain_profile.strikes[strike]

            # Extract raw quote details
            if opt_type == "CE":
                ltp = strike_data.call_ltp
                volume = strike_data.call_volume
                oi = strike_data.call_oi
            else:
                ltp = strike_data.put_ltp
                volume = strike_data.put_volume
                oi = strike_data.put_oi

            if ltp <= 1.0:
                continue

            # Bid/ask estimation if not directly in strike analysis
            bid = round(max(0.05, ltp - 0.5), 2)
            ask = round(ltp + 0.5, 2)
            spread = max(0.2, ask - bid)
            spread_pct = spread / ltp

            # Synthesize symbol if missing
            symbol = f"NSE:{chain_profile.underlying}{int(strike)}{opt_type}"

            # Calculate Greeks
            greeks = self.greeks_engine.calculate_greeks(
                option_type=opt_type,
                strike=strike,
                underlying_futures=fut_price,
                expiry=chain_profile.expiry,
                current_time=current_time,
                market_price=ltp,
            )

            # Estimate premium trajectory
            premium_est = self.cost_model.estimate_premium_trajectory(
                greeks=greeks,
                entry_premium=ltp,
                spot_entry=spot_price,
                spot_target_t1=spot_t1,
                spot_target_t2=spot_t2,
                spot_sl=spot_sl,
                direction=direction.value,
                holding_minutes=hold_mins,
            )

            # Evaluate execution costs and net edge
            costs = self.cost_model.calculate_execution_costs(
                entry_premium=ltp,
                target_premium_t1=premium_est.target_premium_t1,
                stop_loss_premium=premium_est.stop_loss_premium,
                bid_ask_spread=spread,
            )

            # Score the contract (0.0 to 100.0)
            score, reasons = self._score_contract(
                ltp=ltp,
                volume=volume,
                spread_pct=spread_pct,
                greeks=greeks,
                costs=costs,
                holding_minutes=hold_mins,
            )

            if score >= self.min_selection_score:
                contract = SelectedContract(
                    symbol=symbol,
                    underlying=chain_profile.underlying,
                    strike=strike,
                    option_type=opt_type,
                    expiry=chain_profile.expiry,
                    ltp=ltp,
                    bid=bid,
                    ask=ask,
                    spread=spread,
                    spread_pct=spread_pct,
                    volume=volume,
                    oi=oi,
                    greeks=greeks,
                    selection_score=score,
                    selection_reasons=reasons,
                    premium_trajectory=premium_est,
                    execution_costs=costs,
                    holding_minutes=hold_mins,
                )
                evaluated_contracts.append(contract)

        if not evaluated_contracts:
            return None

        # Return the highest-scoring viable contract
        evaluated_contracts.sort(key=lambda c: (c.execution_costs.is_cost_viable, c.selection_score), reverse=True)
        return evaluated_contracts[0]

    def _score_contract(
        self,
        ltp: float,
        volume: int,
        spread_pct: float,
        greeks: OptionGreeks,
        costs: ExecutionCosts,
        holding_minutes: float,
    ) -> tuple[float, List[str]]:
        """Multi-factor scoring heuristic evaluating Greeks, liquidity, and cost viability."""
        score = 0.0
        reasons: List[str] = []
        abs_delta = abs(greeks.delta)

        # 1. Delta Score (up to 35 points)
        if 0.38 <= abs_delta <= 0.52:
            score += 35.0
            reasons.append(f"Ideal ATM/near-ATM Delta ({abs_delta:.2f})")
        elif self.target_delta_min <= abs_delta <= self.target_delta_max:
            score += 28.0
            reasons.append(f"Balanced Delta ({abs_delta:.2f})")
        elif 0.28 <= abs_delta <= 0.65:
            score += 15.0
            reasons.append(f"Acceptable Delta ({abs_delta:.2f})")
        else:
            score += 5.0
            reasons.append(f"Sub-optimal Delta ({abs_delta:.2f})")

        # 2. Liquidity & Spread Friction Score (up to 25 points)
        if spread_pct <= 0.01:
            score += 25.0
            reasons.append("Ultra-tight spread (<1.0%)")
        elif spread_pct <= 0.02:
            score += 18.0
            reasons.append("Tight spread (<2.0%)")
        elif spread_pct <= self.max_acceptable_spread_pct:
            score += 10.0
            reasons.append("Moderate spread")
        else:
            score += 0.0
            reasons.append("Wide spread penalty")

        if volume >= 10000:
            score = min(score + 5.0, 60.0)  # Volume bonus
            reasons.append("High liquidity (Vol >10k)")

        # 3. Theta Cost Budgeting (up to 20 points)
        theta_loss = abs(greeks.theta) * (holding_minutes / 375.0)
        if theta_loss <= 1.0:
            score += 20.0
            reasons.append(f"Minimal theta decay ({theta_loss:.2f} pts/{int(holding_minutes)}m)")
        elif theta_loss <= 2.0:
            score += 14.0
            reasons.append(f"Controlled theta decay ({theta_loss:.2f} pts)")
        elif theta_loss <= 3.5:
            score += 7.0
        else:
            reasons.append("High theta decay cost")

        # 4. Premium Affordability & Leverage Sweet-Spot (up to 20 points)
        if 80.0 <= ltp <= 220.0:
            score += 20.0
            reasons.append(f"Optimal premium sweet-spot (₹{ltp:.1f})")
        elif 60.0 <= ltp <= 320.0:
            score += 14.0
            reasons.append(f"Acceptable premium range (₹{ltp:.1f})")
        elif 40.0 <= ltp <= 450.0:
            score += 8.0
        else:
            reasons.append(f"Extreme premium (₹{ltp:.1f})")

        # Bonus for positive Net R:R viability
        if costs.is_cost_viable:
            score += 5.0
            reasons.append(f"Net R:R viable ({costs.net_rr_t1:.2f} after fees)")

        return round(min(100.0, score), 1), reasons
