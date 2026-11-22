"""Quantitative Option Contract Selector for NIFTY Breakout & Retest Setups.

Selects the single optimal option contract (strike, expiry, CE/PE) based on:
1. Target Delta range (0.35 to 0.55) balancing momentum sensitivity and capital at risk.
2. Order-book liquidity and bid-ask spread friction.
3. Intraday Theta decay budgeting over anticipated holding duration.
4. Premium affordability and leverage sweet-spot.
5. All-in execution cost viability (statutory fees, slippage, and net R:R).
"""

import math
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
    """Evaluates option-chain strikes and selects the highest-scoring liquid contract.

    Strictly separates hard eligibility gates (real broker symbols, observed bid/ask quotes,
    spread tolerance, volume liquidity, valid Greeks, and net R:R viability) from ranking.
    """

    def __init__(
        self,
        greeks_engine: Optional[GreeksEngine] = None,
        cost_model: Optional[PremiumCostModel] = None,
        near_atm_range: int = 5,
        target_delta_min: float = 0.35,
        target_delta_max: float = 0.55,
        min_acceptable_delta: float = 0.25,
        max_acceptable_delta: float = 0.70,
        max_acceptable_spread_pct: float = 0.03,  # Max 3% bid-ask spread
        min_liquidity_volume: int = 500,  # Minimum required trading volume
        default_holding_minutes: float = 30.0,
        min_selection_score: float = 50.0,
    ):
        self.greeks_engine = greeks_engine or GreeksEngine()
        self.cost_model = cost_model or PremiumCostModel()
        self.near_atm_range = near_atm_range
        self.target_delta_min = target_delta_min
        self.target_delta_max = target_delta_max
        self.min_acceptable_delta = min_acceptable_delta
        self.max_acceptable_delta = max_acceptable_delta
        self.max_acceptable_spread_pct = max_acceptable_spread_pct
        self.min_liquidity_volume = min_liquidity_volume
        self.default_holding_minutes = default_holding_minutes
        self.min_selection_score = min_selection_score

    def compute_forward_price(
        self,
        spot_price: float,
        expiry: date,
        current_time: datetime,
        futures_price: Optional[float] = None,
        futures_expiry: Optional[date] = None,
    ) -> float:
        """Computes contract-expiry-matched forward price.

        If a monthly futures price is provided and its expiry matches the contract expiry,
        uses futures_price directly.
        If contract is a weekly option expiring before monthly futures, interpolates the
        cost-of-carry basis so monthly futures basis is not falsely ascribed to a weekly option.
        """
        S = float(spot_price)
        T_opt = self.greeks_engine.compute_time_to_expiry_years(expiry, current_time)
        if T_opt <= 0.0 or S <= 0.0:
            return max(0.05, S)

        r = self.greeks_engine.default_risk_free_rate

        # If no futures price or futures price equals spot, use theoretical forward
        if futures_price is None or abs(futures_price - S) < 0.1:
            return round(S * math.exp(r * T_opt), 2)

        # If futures expiry matches option expiry, use futures price directly
        if futures_expiry and futures_expiry == expiry:
            return float(futures_price)

        # Interpolate cost of carry from monthly futures
        T_fut = (
            self.greeks_engine.compute_time_to_expiry_years(futures_expiry, current_time)
            if futures_expiry
            else max(T_opt, 25.0 / 365.25)
        )
        if T_fut > 0.001 and futures_price > 0.0:
            basis_rate = math.log(futures_price / S) / T_fut
            return round(S * math.exp(basis_rate * T_opt), 2)

        return round(S * math.exp(r * T_opt), 2)

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
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target_1: Optional[float] = None,
        target_2: Optional[float] = None,
        holding_minutes: Optional[float] = None,
    ) -> Optional[SelectedContract]:
        """Selects the single best option contract matching the technical breakout candidate.

        Enforces hard eligibility gates (real broker symbols, observed bid/ask quotes,
        liquidity, Greeks, and net R:R viability) before scoring.
        """
        if not chain_profile or not chain_profile.strikes:
            return None

        current_time = as_of or chain_profile.timestamp
        opt_type = "CE" if direction == BreakoutDirection.BULLISH else "PE"
        hold_mins = holding_minutes or self.default_holding_minutes

        # Determine exact spot entry and price targets
        spot_entry = float(entry_price) if entry_price is not None else float(spot_price)
        if target_1 is not None and stop_loss is not None:
            spot_t1 = float(target_1)
            spot_sl = float(stop_loss)
            if target_2 is not None:
                spot_t2 = float(target_2)
            else:
                spot_t2 = (
                    spot_entry + 2.0 * abs(spot_t1 - spot_entry)
                    if direction == BreakoutDirection.BULLISH
                    else spot_entry - 2.0 * abs(spot_t1 - spot_entry)
                )
        else:
            if direction == BreakoutDirection.BULLISH:
                spot_t1 = spot_entry + target_points_t1
                spot_t2 = spot_entry + target_points_t2
                spot_sl = spot_entry - risk_points
            else:
                spot_t1 = spot_entry - target_points_t1
                spot_t2 = spot_entry - target_points_t2
                spot_sl = spot_entry + risk_points

        # Contract-expiry-matched forward price
        fwd_price = self.compute_forward_price(
            spot_price=spot_entry,
            expiry=chain_profile.expiry,
            current_time=current_time,
            futures_price=futures_price,
        )

        # Filter near-ATM strikes
        sorted_strikes = sorted(chain_profile.strikes.keys())
        sorted_by_dist = sorted(sorted_strikes, key=lambda k: abs(k - spot_entry))
        candidate_strikes = sorted(sorted_by_dist[: 2 * self.near_atm_range + 1])

        evaluated_contracts: List[SelectedContract] = []

        for strike in candidate_strikes:
            strike_data = chain_profile.strikes[strike]

            # Extract authentic broker quotes and metadata (NEVER invent symbols or bid/ask)
            if opt_type == "CE":
                symbol = strike_data.call_symbol
                ltp = strike_data.call_ltp
                volume = strike_data.call_volume
                oi = strike_data.call_oi
                bid = strike_data.call_bid
                ask = strike_data.call_ask
                broker_greeks = {
                    "iv": strike_data.call_iv,
                    "delta": strike_data.call_delta,
                    "gamma": strike_data.call_gamma,
                    "theta": strike_data.call_theta,
                    "vega": strike_data.call_vega,
                }
            else:
                symbol = strike_data.put_symbol
                ltp = strike_data.put_ltp
                volume = strike_data.put_volume
                oi = strike_data.put_oi
                bid = strike_data.put_bid
                ask = strike_data.put_ask
                broker_greeks = {
                    "iv": strike_data.put_iv,
                    "delta": strike_data.put_delta,
                    "gamma": strike_data.put_gamma,
                    "theta": strike_data.put_theta,
                    "vega": strike_data.put_vega,
                }

            # HARD ELIGIBILITY GATE 1: Real exchange symbol required
            if not symbol or not symbol.startswith("NSE:"):
                continue

            # HARD ELIGIBILITY GATE 2: Valid, observed broker bid and ask quotes
            if bid is None or ask is None or bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue

            spread = round(ask - bid, 2)
            spread_pct = round(spread / ask, 4)

            # HARD ELIGIBILITY GATE 3: Spread friction limit
            if spread_pct > self.max_acceptable_spread_pct:
                continue

            # HARD ELIGIBILITY GATE 4: Liquidity volume threshold
            if volume < self.min_liquidity_volume:
                continue

            # Effective market price for Greeks solver
            mkt_price = ltp if ltp > 0.05 else ask

            # Calculate Greeks with authentic broker fallback and contract-matched forward
            greeks = self.greeks_engine.calculate_greeks(
                option_type=opt_type,
                strike=strike,
                underlying_futures=fwd_price,
                expiry=chain_profile.expiry,
                current_time=current_time,
                market_price=mkt_price,
                broker_greeks=broker_greeks,
            )

            # HARD ELIGIBILITY GATE 5: Valid Greeks and Delta limits
            if not greeks.is_calculated and greeks.model_used != "FYERS":
                continue
            if greeks.iv <= 0.001:
                continue
            if not (self.min_acceptable_delta <= abs(greeks.delta) <= self.max_acceptable_delta):
                continue

            # Realistic entry premium: Buyer buys on the Ask price
            entry_premium = ask

            # Estimate option premium trajectory based on exact technical entry/targets
            premium_est = self.cost_model.estimate_premium_trajectory(
                greeks=greeks,
                entry_premium=entry_premium,
                spot_entry=spot_entry,
                spot_target_t1=spot_t1,
                spot_target_t2=spot_t2,
                spot_sl=spot_sl,
                direction=direction.value,
                holding_minutes=hold_mins,
            )

            # Evaluate exact asymmetric Indian execution costs
            costs = self.cost_model.calculate_execution_costs(
                entry_premium=entry_premium,
                target_premium_t1=premium_est.target_premium_t1,
                stop_loss_premium=premium_est.stop_loss_premium,
                bid_ask_spread=spread,
                underlying=chain_profile.underlying,
            )

            # HARD ELIGIBILITY GATE 6: Net Reward-to-Risk viability
            if not costs.is_cost_viable:
                continue

            # Score the eligible contract (0.0 to 100.0)
            score, reasons = self._score_contract(
                ltp=mkt_price,
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
                    ltp=round(mkt_price, 2),
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

        # Return the highest-scoring eligible contract
        evaluated_contracts.sort(key=lambda c: c.selection_score, reverse=True)
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
