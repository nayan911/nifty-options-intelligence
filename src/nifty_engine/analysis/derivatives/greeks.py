"""Option Greeks and Implied Volatility computation engine for NIFTY derivatives.

Provides Black-76 / Black-Scholes analytical Greeks computation using futures price
as underlying, dynamic Newton-Raphson IV solver, and transparent fallback from broker-reported Greeks.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, Optional

import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


@dataclass
class OptionGreeks:
    """Computed or broker-supplied Greeks and volatility metrics."""

    iv: float  # Annualized volatility as decimal (e.g. 0.15 = 15%)
    delta: float  # First-order spot sensitivity
    gamma: float  # Second-order spot sensitivity
    theta: float  # Daily theta decay (negative points per calendar day)
    vega: float  # Premium change per 1% change in IV (e.g. 15% -> 16%)
    rho: float  # Sensitivity to 1% change in risk-free interest rate
    is_calculated: bool = False
    model_used: str = "FYERS"  # "FYERS" or "BLACK_76"


class GreeksEngine:
    """Computes Black-76 option pricing, exact Greeks, and implied volatility.

    In Indian derivatives markets, index options settle against underlying index spot,
    but actively trade against index futures (which naturally embed dividend yield
    and cost of carry). Black-76 with futures price $F$ provides the standard valuation model.
    """

    def __init__(
        self,
        default_risk_free_rate: float = 0.065,  # 6.5% typical 91-day T-Bill rate
        trading_days_per_year: float = 365.25,
        market_close_time: time = time(15, 30, 0),
    ):
        self.default_risk_free_rate = default_risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.market_close_time = market_close_time

    def compute_time_to_expiry_years(
        self,
        expiry: date,
        current_time: datetime,
    ) -> float:
        """Computes continuous time-to-expiry in annual fractions.

        Measures exact remaining time up to market close (15:30 IST) on the expiry date.
        """
        if current_time.tzinfo is None:
            current_time = IST.localize(current_time)
        else:
            current_time = current_time.astimezone(IST)

        expiry_dt = datetime.combine(expiry, self.market_close_time)
        expiry_dt = IST.localize(expiry_dt)

        seconds_remaining = (expiry_dt - current_time).total_seconds()
        if seconds_remaining <= 0:
            return 0.0001  # Minimum time buffer (prevents division by zero at settlement)

        return seconds_remaining / (self.trading_days_per_year * 86400.0)

    def calculate_black76_price(
        self,
        option_type: str,
        underlying_futures: float,
        strike: float,
        time_to_expiry_years: float,
        volatility: float,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculates theoretical Black-76 option price for futures-settled option."""
        opt_type = option_type.upper().strip()
        F = float(underlying_futures)
        K = float(strike)
        T = max(0.0001, float(time_to_expiry_years))
        r = float(risk_free_rate if risk_free_rate is not None else self.default_risk_free_rate)
        sigma = max(0.001, float(volatility))

        if F <= 0.0 or K <= 0.0:
            return 0.0

        discount = math.exp(-r * T)
        denom = sigma * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * (sigma**2) * T) / denom
        d2 = d1 - denom

        if opt_type in ("CE", "CALL"):
            price = discount * (F * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            price = discount * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

        return max(0.0, price)

    def solve_implied_volatility(
        self,
        option_type: str,
        market_price: float,
        underlying_futures: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: Optional[float] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
    ) -> float:
        """Solves Implied Volatility (IV) from option market price using Newton-Raphson.

        Falls back to bounded bisection if derivative vanishes or gradient diverges.
        """
        opt_type = option_type.upper().strip()
        F = float(underlying_futures)
        K = float(strike)
        T = max(0.0001, float(time_to_expiry_years))
        r = float(risk_free_rate if risk_free_rate is not None else self.default_risk_free_rate)
        mkt = max(0.05, float(market_price))

        discount = math.exp(-r * T)
        intrinsic = (
            max(0.0, discount * (F - K))
            if opt_type in ("CE", "CALL")
            else max(0.0, discount * (K - F))
        )
        if mkt <= intrinsic:
            return 0.01  # Boundary minimum 1% IV

        # Initial estimate
        sigma = 0.18  # 18% starting volatility

        for _ in range(max_iterations):
            denom = sigma * math.sqrt(T)
            d1 = (math.log(F / K) + 0.5 * (sigma**2) * T) / denom
            d2 = d1 - denom

            if opt_type in ("CE", "CALL"):
                price = discount * (F * norm_cdf(d1) - K * norm_cdf(d2))
            else:
                price = discount * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

            diff = price - mkt
            if abs(diff) < tolerance:
                return round(sigma, 4)

            # Vega = dPrice / dSigma
            vega = F * discount * norm_pdf(d1) * math.sqrt(T)
            if vega < 1e-6:
                break  # Flat derivative, switch to bisection

            step = diff / vega
            sigma -= step
            if sigma <= 0.001 or sigma >= 4.0:
                break

        # Robust bisection fallback
        low_vol = 0.01
        high_vol = 3.5
        for _ in range(40):
            mid_vol = 0.5 * (low_vol + high_vol)
            denom = mid_vol * math.sqrt(T)
            d1 = (math.log(F / K) + 0.5 * (mid_vol**2) * T) / denom
            d2 = d1 - denom

            if opt_type in ("CE", "CALL"):
                price = discount * (F * norm_cdf(d1) - K * norm_cdf(d2))
            else:
                price = discount * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

            diff = price - mkt
            if abs(diff) < tolerance:
                return round(mid_vol, 4)

            if diff > 0:
                high_vol = mid_vol
            else:
                low_vol = mid_vol

        return round(0.5 * (low_vol + high_vol), 4)

    def calculate_greeks(
        self,
        option_type: str,
        strike: float,
        underlying_futures: float,
        expiry: date,
        current_time: datetime,
        market_price: float,
        risk_free_rate: Optional[float] = None,
        broker_greeks: Optional[Dict[str, Any]] = None,
        broker_greeks_timestamp: Optional[datetime] = None,
        quote_timestamp: Optional[datetime] = None,
        max_greeks_age_seconds: Optional[float] = 30.0,
    ) -> OptionGreeks:
        """Generates complete OptionGreeks, preferring valid, fresh broker Greeks and falling back to Black-76.

        Validates freshness and correspondence of broker Greeks:
        - If broker Greeks are older than max_greeks_age_seconds, or future-dated beyond clock skew,
          or if refreshed quote timestamp indicates the broker Greeks are from a stale snapshot,
          the engine calculates fresh analytical Black-76 Greeks using the current market price.
        """
        opt_type = option_type.upper().strip()
        F = float(underlying_futures)
        K = float(strike)
        mkt = float(market_price)
        r = float(risk_free_rate if risk_free_rate is not None else self.default_risk_free_rate)
        T = self.compute_time_to_expiry_years(expiry=expiry, current_time=current_time)

        # 1. Primary check: Use broker Greeks if complete, valid, non-zero, and fresh
        if broker_greeks:
            b_iv = broker_greeks.get("iv")
            b_delta = broker_greeks.get("delta")
            b_gamma = broker_greeks.get("gamma")
            b_theta = broker_greeks.get("theta")
            b_vega = broker_greeks.get("vega")

            # Validate completeness and mathematical validity:
            # - IV must be positive (> 0.01)
            # - Delta must be non-trivial (|delta| > 0.01)
            # - Gamma must be present and positive (> 0.0)
            # - Theta must be present and non-zero (!= 0.0)
            # - Vega must be present and positive (> 0.0)
            # Missing or zero gamma/theta/vega must NOT be defaulted to 0.0;
            # instead, fall back to calculating a consistent Black-76 set.
            is_complete = (
                b_iv is not None
                and b_delta is not None
                and b_gamma is not None
                and b_theta is not None
                and b_vega is not None
            )
            is_valid = False
            if is_complete:
                try:
                    f_iv = float(b_iv)
                    f_delta = float(b_delta)
                    f_gamma = float(b_gamma)
                    f_theta = float(b_theta)
                    f_vega = float(b_vega)
                    f_rho = float(broker_greeks.get("rho") or 0.0)

                    # 1. Require strictly finite numeric values (rejects NaN, +Inf, -Inf)
                    is_finite = (
                        math.isfinite(f_iv)
                        and math.isfinite(f_delta)
                        and math.isfinite(f_gamma)
                        and math.isfinite(f_theta)
                        and math.isfinite(f_vega)
                        and math.isfinite(f_rho)
                    )

                    # 2. Domain constraints applied strictly on verified finite numbers
                    is_valid = (
                        is_finite
                        and f_iv > 0.01
                        and 0.01 < abs(f_delta) <= 1.05
                        and f_gamma > 0.0
                        and f_theta != 0.0
                        and f_vega > 0.0
                    )
                except (ValueError, TypeError):
                    is_valid = False

            if is_valid:
                bg_ts = (
                    broker_greeks_timestamp
                    or broker_greeks.get("timestamp")
                    or broker_greeks.get("quote_time")
                )
                max_age = (
                    max_greeks_age_seconds
                    if max_greeks_age_seconds is not None
                    else 30.0
                )
                is_fresh = True
                if bg_ts is not None:
                    greeks_age = (current_time - bg_ts).total_seconds()
                    # Reject stale or materially future-dated broker Greeks
                    if greeks_age > max_age or greeks_age < -2.0:
                        is_fresh = False
                    # When quote timestamp is refreshed, ensure broker Greeks correspond to that quote
                    elif quote_timestamp is not None:
                        drift = (quote_timestamp - bg_ts).total_seconds()
                        if drift > max_age:
                            is_fresh = False
                elif quote_timestamp is not None:
                    # Provenance gating: If quote timestamp exists but broker Greeks omit a timestamp,
                    # do NOT assume freshness from quote time without guaranteed provenance
                    is_fresh = False

                if is_fresh:
                    return OptionGreeks(
                        iv=round(f_iv / 100.0 if f_iv > 1.0 else f_iv, 4),
                        delta=round(f_delta, 4),
                        gamma=round(f_gamma, 6),
                        theta=round(f_theta, 3),
                        vega=round(f_vega, 3),
                        rho=round(f_rho, 4),
                        is_calculated=False,
                        model_used="FYERS",
                    )
                else:
                    logger.debug(
                        "Broker Greeks for strike %.1f rejected (stale/unmatched: bg_ts=%s, quote_ts=%s, as_of=%s). Falling back to Black-76.",
                        K,
                        bg_ts,
                        quote_timestamp,
                        current_time,
                    )
            else:
                logger.debug(
                    "Broker Greeks for strike %.1f incomplete or invalid (iv=%s, delta=%s, gamma=%s, theta=%s, vega=%s). Falling back to Black-76.",
                    K,
                    b_iv,
                    b_delta,
                    b_gamma,
                    b_theta,
                    b_vega,
                )

        # 2. Analytical Black-76 fallback
        if F <= 0.0 or K <= 0.0 or mkt <= 0.0:
            return OptionGreeks(
                iv=0.0,
                delta=0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                is_calculated=True,
                model_used="BLACK_76",
            )

        # Calculate IV
        sigma = self.solve_implied_volatility(
            option_type=opt_type,
            market_price=mkt,
            underlying_futures=F,
            strike=K,
            time_to_expiry_years=T,
            risk_free_rate=r,
        )

        discount = math.exp(-r * T)
        denom = sigma * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * (sigma**2) * T) / denom
        d2 = d1 - denom

        # Delta
        if opt_type in ("CE", "CALL"):
            delta = discount * norm_cdf(d1)
        else:
            delta = -discount * norm_cdf(-d1)

        # Gamma (identical for Call and Put)
        gamma = (discount * norm_pdf(d1)) / (F * denom)

        # Vega (change per 1% IV change = 0.01 * dPrice/dSigma)
        vega = (F * discount * norm_pdf(d1) * math.sqrt(T)) / 100.0

        # Theta per calendar day (1/365 of annual theta)
        # Note: Theta decay is an expense for option buyers, conventionally reported as negative points/day
        common_decay = -(F * discount * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        if opt_type in ("CE", "CALL"):
            annual_theta = common_decay - r * K * discount * norm_cdf(d2) + r * F * discount * norm_cdf(d1)
        else:
            annual_theta = common_decay + r * K * discount * norm_cdf(-d2) - r * F * discount * norm_cdf(-d1)
        daily_theta = annual_theta / 365.25

        # Rho per 1% change in interest rate
        if opt_type in ("CE", "CALL"):
            annual_rho = -T * discount * (F * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            annual_rho = -T * discount * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        rho = annual_rho / 100.0

        return OptionGreeks(
            iv=round(sigma, 4),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(daily_theta, 3),
            vega=round(vega, 3),
            rho=round(rho, 4),
            is_calculated=True,
            model_used="BLACK_76",
        )
