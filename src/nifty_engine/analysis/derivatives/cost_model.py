"""Option Premium Movement and Complete Indian Execution Cost Model.

Implements:
1. Second-order Taylor series expansion (Delta, Gamma, Theta, Vega) for premium trajectory.
2. Complete statutory Indian transaction cost calculation (FYERS brokerage, STT 0.15%,
   Exchange turnover fees, SEBI charges, Stamp Duty, GST 18%, and Bid-Ask slippage).
3. Rigorous Net Reward-to-Risk filtering ensuring edge is not eroded by market friction.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from nifty_engine.analysis.derivatives.greeks import OptionGreeks

DEFAULT_LOT_SIZES: Dict[str, int] = {
    "NIFTY": 65,
    "BANKNIFTY": 15,
    "SENSEX": 10,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
}


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
    stt_target: float  # Securities Transaction Tax (0.15% on target sell turnover)
    stt_sl: float  # Securities Transaction Tax (0.15% on SL sell turnover)
    exchange_fee_target: float  # NSE 0.0503% on premium turnover at target exit
    exchange_fee_sl: float  # NSE 0.0503% on premium turnover at SL exit
    sebi_charges: float  # ₹10 / crore (0.0001%) on premium turnover
    stamp_duty: float  # 0.003% on buy turnover
    gst_target: float  # 18% on (brokerage + exchange fee + sebi charges) at target
    gst_sl: float  # 18% on (brokerage + exchange fee + sebi charges) at SL
    slippage_points: float  # Slippage friction in option premium points
    spread_cost_points: float  # Bid-ask spread cost in option premium points
    total_statutory_target_rupees: float  # Total government + exchange fees in INR at target
    total_statutory_sl_rupees: float  # Total government + exchange fees in INR at SL
    target_friction_points: float  # Total friction points at target exit
    sl_friction_points: float  # Total friction points at SL exit
    gross_reward_points_t1: float  # Target 1 gross premium gain
    gross_risk_points: float  # Stop Loss gross premium risk
    gross_rr_t1: float  # Gross Reward-to-Risk ratio
    net_reward_points_t1: float  # Net Reward after friction
    net_risk_points: float  # Net Risk including friction
    net_rr_t1: float  # Net Reward-to-Risk ratio
    is_cost_viable: bool  # True if net_rr_t1 >= min_net_rr threshold
    lot_size: int = 65

    @property
    def total_statutory_rupees(self) -> float:
        """Backward-compatible total statutory fees at target exit."""
        return self.total_statutory_target_rupees

    @property
    def total_friction_rupees(self) -> float:
        """Backward-compatible total friction in INR per lot at target exit."""
        return round(self.target_friction_points * self.lot_size, 2)

    @property
    def total_friction_points(self) -> float:
        """Backward-compatible total friction in points at target exit."""
        return self.target_friction_points

    @property
    def stt(self) -> float:
        """Backward-compatible STT at target exit."""
        return self.stt_target

    @property
    def exchange_turnover_fee(self) -> float:
        """Backward-compatible exchange turnover fee at target exit."""
        return self.exchange_fee_target

    @property
    def gst(self) -> float:
        """Backward-compatible GST at target exit."""
        return self.gst_target


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

    def get_lot_size(self, underlying: Optional[str] = None) -> int:
        """Resolves official lot size by underlying instrument."""
        if underlying:
            u = underlying.upper()
            if "BANKNIFTY" in u or "NIFTYBANK" in u:
                return DEFAULT_LOT_SIZES.get("BANKNIFTY", 15)
            elif "NIFTY" in u:
                return DEFAULT_LOT_SIZES.get("NIFTY", 65)
            elif "SENSEX" in u:
                return DEFAULT_LOT_SIZES.get("SENSEX", 10)
            elif "FINNIFTY" in u:
                return DEFAULT_LOT_SIZES.get("FINNIFTY", 65)
            elif "MIDCPNIFTY" in u:
                return DEFAULT_LOT_SIZES.get("MIDCPNIFTY", 120)
        return self.lot_size

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

        # Intraday theta cost:
        # Standard trading year = 250 days; Indian session = 375 minutes (09:15 to 15:30 IST).
        # Black-76 theta is conventionally computed per calendar day (1/365.25 of annual).
        # We scale calendar theta to trading-day theta (365.25 / 250.0) so scaling by (holding_minutes / 375.0)
        # represents exact intraday minute-by-minute session time decay.
        trading_day_theta = abs(greeks.theta) * (365.25 / 250.0) if greeks.theta is not None else 0.0
        time_frac = max(0.0, min(1.0, holding_minutes / 375.0))
        theta_decay = round(trading_day_theta * time_frac, 2)

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
        underlying: Optional[str] = None,
    ) -> ExecutionCosts:
        """Calculates comprehensive transaction fees, slippage, and net R:R metrics

        with separate, exact Target exit and Stop-Loss exit cost scenarios.
        """
        lot = lot_size or self.get_lot_size(underlying)
        p_buy = max(0.05, float(entry_premium))
        p_target = max(0.05, float(target_premium_t1))
        p_sl = max(0.05, float(stop_loss_premium))

        # Gross metrics in premium points
        gross_reward_pts = max(0.0, p_target - p_buy)
        gross_risk_pts = max(0.5, p_buy - p_sl)
        gross_rr = round(gross_reward_pts / gross_risk_pts, 2)

        # 1. Statutory Costs Calculation
        buy_turnover = p_buy * lot
        sell_target_turnover = p_target * lot
        sell_sl_turnover = p_sl * lot

        brokerage = 2.0 * self.flat_brokerage_per_order  # ₹20 buy + ₹20 sell = ₹40.0
        stamp_duty = round(self.stamp_duty_buy_rate * buy_turnover, 2)

        # Target Scenario (evaluated at Target 1 exit)
        total_target_turnover = buy_turnover + sell_target_turnover
        stt_target = round(self.stt_sell_rate * sell_target_turnover, 2)
        exchange_fees_target = round(self.exchange_fee_rate * total_target_turnover, 2)
        sebi_charges_target = round(self.sebi_rate * total_target_turnover, 2)
        gst_target = round(self.gst_rate * (brokerage + exchange_fees_target + sebi_charges_target), 2)
        total_statutory_target = round(
            brokerage + stt_target + exchange_fees_target + sebi_charges_target + stamp_duty + gst_target, 2
        )

        # Stop-Loss Scenario (evaluated at Stop-Loss exit)
        total_sl_turnover = buy_turnover + sell_sl_turnover
        stt_sl = round(self.stt_sell_rate * sell_sl_turnover, 2)
        exchange_fees_sl = round(self.exchange_fee_rate * total_sl_turnover, 2)
        sebi_charges_sl = round(self.sebi_rate * total_sl_turnover, 2)
        gst_sl = round(self.gst_rate * (brokerage + exchange_fees_sl + sebi_charges_sl), 2)
        total_statutory_sl = round(
            brokerage + stt_sl + exchange_fees_sl + sebi_charges_sl + stamp_duty + gst_sl, 2
        )

        # 2. Execution Friction: Bid-Ask Spread and Slippage
        slip_pts = self.default_slippage_points if slippage_points is None else float(slippage_points)
        spread_pts = max(0.05, float(bid_ask_spread))
        spread_friction_pts = spread_pts

        target_friction_pts = round((total_statutory_target / lot) + spread_friction_pts + (2.0 * slip_pts), 2)
        sl_friction_pts = round((total_statutory_sl / lot) + spread_friction_pts + (2.0 * slip_pts), 2)

        # 3. Net Metrics
        net_reward_pts = max(0.0, gross_reward_pts - target_friction_pts)
        net_risk_pts = gross_risk_pts + sl_friction_pts
        net_rr = round(net_reward_pts / net_risk_pts, 2) if net_risk_pts > 0 else 0.0

        is_cost_viable = bool(net_rr >= self.min_net_rr and net_reward_pts > 0.0)

        return ExecutionCosts(
            brokerage=brokerage,
            stt_target=stt_target,
            stt_sl=stt_sl,
            exchange_fee_target=exchange_fees_target,
            exchange_fee_sl=exchange_fees_sl,
            sebi_charges=sebi_charges_target,
            stamp_duty=stamp_duty,
            gst_target=gst_target,
            gst_sl=gst_sl,
            slippage_points=slip_pts,
            spread_cost_points=round(spread_friction_pts, 2),
            total_statutory_target_rupees=total_statutory_target,
            total_statutory_sl_rupees=total_statutory_sl,
            target_friction_points=target_friction_pts,
            sl_friction_points=sl_friction_pts,
            gross_reward_points_t1=round(gross_reward_pts, 2),
            gross_risk_points=round(gross_risk_pts, 2),
            gross_rr_t1=gross_rr,
            net_reward_points_t1=round(net_reward_pts, 2),
            net_risk_points=round(net_risk_pts, 2),
            net_rr_t1=net_rr,
            is_cost_viable=is_cost_viable,
            lot_size=lot,
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
