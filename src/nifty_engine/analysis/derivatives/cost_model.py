"""Option Premium Movement and Complete Indian Execution Cost Model.

Implements:
1. Second-order Taylor series expansion (Delta, Gamma, Theta, Vega) for premium trajectory.
2. Complete statutory Indian transaction cost calculation (FYERS brokerage, STT 0.15%,
   Exchange turnover fees, SEBI charges, Stamp Duty, GST 18%, and Bid-Ask slippage).
3. Rigorous Net Reward-to-Risk filtering ensuring edge is not eroded by market friction.
"""

import math
from dataclasses import dataclass
from typing import Optional

from nifty_engine.analysis.derivatives.greeks import OptionGreeks


@dataclass
class PremiumEstimate:
    """Estimated option premium trajectory under technical price targets."""

    initial_premium: float
    target_premium_t1: float
    target_premium_t2: float
    stop_loss_premium: float
    delta_contribution_t1: float
    gamma_contribution_t1: float
    theta_cost: float
    vega_contribution: float
    required_spot_move_t1: float
    confidence: str = "HIGH"


@dataclass
class ExecutionCosts:
    """Detailed breakdown of Indian statutory fees, broker charges, and execution friction."""

    brokerage: float  # FYERS flat ₹20 entry + ₹20 exit = ₹40.0
    stt: float  # Securities Transaction Tax (0.15% on sell turnover)
    exchange_turnover_fee: float  # NSE 0.0503% on premium turnover (buy + sell)
    sebi_charges: float  # ₹10 / crore (0.0001%) on premium turnover
    stamp_duty: float  # 0.003% on buy turnover
    gst: float  # 18% on (brokerage + exchange fee + sebi charges)
    slippage_points: float  # Slippage friction in option premium points
    spread_cost_points: float  # Half-spread cost in option premium points
    total_statutory_rupees: float  # Total government + exchange fees in INR per lot
    total_friction_rupees: float  # Total all-in cost (statutory + slippage + spread) in INR per lot
    total_friction_points: float  # Total friction converted to option premium points
    gross_reward_points_t1: float  # Target 1 gross premium gain
    gross_risk_points: float  # Stop Loss gross premium risk
    gross_rr_t1: float  # Gross Reward-to-Risk ratio
    net_reward_points_t1: float  # Net Reward after friction
    net_risk_points: float  # Net Risk including friction
    net_rr_t1: float  # Net Reward-to-Risk ratio
    is_cost_viable: bool  # True if net_rr_t1 >= min_net_rr threshold


class PremiumCostModel:
    """Models option premium movement and evaluates exact Indian statutory friction."""

    def __init__(
        self,
        lot_size: int = 65,  # Standard NIFTY options lot size
        flat_brokerage_per_order: float = 20.0,  # FYERS ₹20 per executed order
        stt_sell_rate: float = 0.0015,  # 0.15% STT on option sell turnover (Finance Act 2024)
        exchange_fee_rate: float = 0.000503,  # NSE 0.0503% turnover fee
        sebi_rate: float = 0.000001,  # ₹10 per crore
        stamp_duty_buy_rate: float = 0.00003,  # 0.003% on buy
        gst_rate: float = 0.18,  # 18% GST
        default_slippage_points: float = 0.5,  # 0.5 point execution slippage
        min_net_rr: float = 1.3,  # Minimum net reward-to-risk threshold
    ):
        self.lot_size = lot_size
        self.flat_brokerage_per_order = flat_brokerage_per_order
        self.stt_sell_rate = stt_sell_rate
        self.exchange_fee_rate = exchange_fee_rate
        self.sebi_rate = sebi_rate
        self.stamp_duty_buy_rate = stamp_duty_buy_rate
        self.gst_rate = gst_rate
        self.default_slippage_points = default_slippage_points
        self.min_net_rr = min_net_rr

    def estimate_premium_trajectory(
        self,
        greeks: OptionGreeks,
        entry_premium: float,
        spot_entry: float,
        spot_target_t1: float,
        spot_target_t2: float,
        spot_sl: float,
        direction: str,  # "BULLISH" or "BEARISH"
        holding_minutes: float = 30.0,
        iv_change_pts: float = 0.0,
    ) -> PremiumEstimate:
        """Estimates option premium movement using second-order Taylor expansion."""
        p_entry = max(0.05, float(entry_premium))

        # Signed spot price move
        # For Bullish (Call): delta > 0, spot increases -> premium increases
        # For Bearish (Put): delta < 0, spot decreases -> (spot_target - spot_entry) < 0 -> delta * move > 0 -> premium increases!
        ds_t1 = spot_target_t1 - spot_entry
        ds_t2 = spot_target_t2 - spot_entry
        ds_sl = spot_sl - spot_entry

        # Intraday theta cost fraction (standard trading day is 375 minutes: 09:15 to 15:30)
        time_frac = max(0.0, min(1.0, holding_minutes / 375.0))
        theta_decay = abs(greeks.theta) * time_frac  # points lost to time

        # Vega contribution
        vega_pnl = greeks.vega * iv_change_pts

        # 1. Target 1 trajectory
        delta_pnl_t1 = greeks.delta * ds_t1
        gamma_pnl_t1 = 0.5 * greeks.gamma * (ds_t1**2)
        dp_t1 = delta_pnl_t1 + gamma_pnl_t1 - theta_decay + vega_pnl
        p_t1 = max(0.05, p_entry + dp_t1)

        # 2. Target 2 trajectory
        delta_pnl_t2 = greeks.delta * ds_t2
        gamma_pnl_t2 = 0.5 * greeks.gamma * (ds_t2**2)
        dp_t2 = delta_pnl_t2 + gamma_pnl_t2 - theta_decay + vega_pnl
        p_t2 = max(0.05, p_entry + dp_t2)

        # 3. Stop loss trajectory
        delta_pnl_sl = greeks.delta * ds_sl
        gamma_pnl_sl = 0.5 * greeks.gamma * (ds_sl**2)
        dp_sl = delta_pnl_sl + gamma_pnl_sl - theta_decay + vega_pnl
        # Stop loss cannot exceed entry premium
        p_sl = max(0.05, min(p_entry, p_entry + dp_sl))

        return PremiumEstimate(
            initial_premium=round(p_entry, 2),
            target_premium_t1=round(p_t1, 2),
            target_premium_t2=round(p_t2, 2),
            stop_loss_premium=round(p_sl, 2),
            delta_contribution_t1=round(delta_pnl_t1, 2),
            gamma_contribution_t1=round(gamma_pnl_t1, 2),
            theta_cost=round(theta_decay, 2),
            vega_contribution=round(vega_pnl, 2),
            required_spot_move_t1=round(abs(ds_t1), 2),
            confidence="HIGH" if greeks.iv > 0.05 else "ESTIMATED",
        )

    def calculate_execution_costs(
        self,
        entry_premium: float,
        target_premium_t1: float,
        stop_loss_premium: float,
        bid_ask_spread: float = 0.5,
        slippage_points: Optional[float] = None,
        lot_size: Optional[int] = None,
    ) -> ExecutionCosts:
        """Calculates comprehensive transaction fees, slippage, and net R:R metrics."""
        lot = lot_size or self.lot_size
        p_buy = max(0.05, float(entry_premium))
        p_target = max(0.05, float(target_premium_t1))
        p_sl = max(0.05, float(stop_loss_premium))

        # Gross metrics in premium points
        gross_reward_pts = max(0.0, p_target - p_buy)
        gross_risk_pts = max(0.5, p_buy - p_sl)
        gross_rr = round(gross_reward_pts / gross_risk_pts, 2)

        # 1. Statutory Costs Calculation (Evaluated at Target exit for viable trade)
        buy_turnover = p_buy * lot
        sell_turnover = p_target * lot
        total_turnover = buy_turnover + sell_turnover

        # Brokerage: ₹20 buy + ₹20 sell = ₹40.0
        brokerage = 2.0 * self.flat_brokerage_per_order

        # STT: 0.15% on sell turnover
        stt = round(self.stt_sell_rate * sell_turnover, 2)

        # Exchange turnover fees: 0.0503% on buy + sell turnover
        exchange_fees = round(self.exchange_fee_rate * total_turnover, 2)

        # SEBI turnover charges: ₹10 / crore (0.0001%)
        sebi_charges = round(self.sebi_rate * total_turnover, 2)

        # Stamp duty: 0.003% on buy turnover
        stamp_duty = round(self.stamp_duty_buy_rate * buy_turnover, 2)

        # GST: 18% on (brokerage + exchange fees + SEBI charges)
        gst = round(self.gst_rate * (brokerage + exchange_fees + sebi_charges), 2)

        total_statutory_rupees = round(brokerage + stt + exchange_fees + sebi_charges + stamp_duty + gst, 2)

        # 2. Execution Friction: Bid-Ask Spread and Slippage
        slip_pts = self.default_slippage_points if slippage_points is None else float(slippage_points)
        spread_pts = max(0.1, float(bid_ask_spread))
        # Crossing the spread on entry and exit costs half-spread each = 1 full spread
        spread_friction_pts = spread_pts

        statutory_pts = total_statutory_rupees / lot
        total_friction_pts = round(statutory_pts + spread_friction_pts + (2.0 * slip_pts), 2)
        total_friction_rupees = round(total_friction_pts * lot, 2)

        # 3. Net Metrics
        net_reward_pts = max(0.0, gross_reward_pts - total_friction_pts)
        net_risk_pts = gross_risk_pts + total_friction_pts
        net_rr = round(net_reward_pts / net_risk_pts, 2) if net_risk_pts > 0 else 0.0

        is_cost_viable = bool(net_rr >= self.min_net_rr and net_reward_pts > 0.0)

        return ExecutionCosts(
            brokerage=brokerage,
            stt=stt,
            exchange_turnover_fee=exchange_fees,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            slippage_points=slip_pts,
            spread_cost_points=round(spread_friction_pts, 2),
            total_statutory_rupees=total_statutory_rupees,
            total_friction_rupees=total_friction_rupees,
            total_friction_points=total_friction_pts,
            gross_reward_points_t1=round(gross_reward_pts, 2),
            gross_risk_points=round(gross_risk_pts, 2),
            gross_rr_t1=gross_rr,
            net_reward_points_t1=round(net_reward_pts, 2),
            net_risk_points=round(net_risk_pts, 2),
            net_rr_t1=net_rr,
            is_cost_viable=is_cost_viable,
        )

    def calculate_required_spot_move_for_premium_target(
        self,
        greeks: OptionGreeks,
        target_premium_gain: float,
    ) -> float:
        """Solves for spot index move $\\Delta S$ needed to achieve target premium gain.

        Solves quadratic equation: $0.5 \\Gamma (\\Delta S)^2 + |\\Delta| \\Delta S - \\text{target} = 0$.
        """
        target = float(target_premium_gain)
        a = 0.5 * greeks.gamma
        b = abs(greeks.delta)
        c = -target

        if b <= 0.0:
            return float("inf")

        if a <= 1e-7:
            return round(target / b, 2)

        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0:
            return float("inf")

        delta_s = (-b + math.sqrt(discriminant)) / (2.0 * a)
        return round(max(0.0, delta_s), 2)
