"""
Cash-secured put yield calculator.

Selling (writing) a cash-secured put means you set aside ``strike * 100`` in
cash per contract and collect the option premium up front. If the stock stays
above the strike until expiry you keep the premium; otherwise you buy the
shares at the strike price (your effective cost basis is ``strike - premium``).

The annualized yield on the collateral is::

    yield = (premium / strike) * (365 / days_to_expiry)

This module prices puts with the Black-Scholes model and searches for strikes
whose premium generates a target annualized yield, e.g. 8% to 10%.

Reference: https://www.investopedia.com/terms/c/cashsecuredput.asp

Note: this is an educational model, not financial advice. Real premiums come
from the market (bid price), and the trade-off for a higher yield is a strike
closer to the current price, i.e. a higher chance of being assigned.
"""

from __future__ import annotations

from math import erf, exp, log, sqrt
from typing import NamedTuple

DAYS_PER_YEAR = 365


class PutCandidate(NamedTuple):
    strike: float
    premium: float
    annualized_yield: float
    probability_otm: float
    breakeven: float


def _normal_cdf(x: float) -> float:
    """
    Cumulative distribution function of the standard normal distribution.

    >>> _normal_cdf(0.0)
    0.5
    >>> round(_normal_cdf(1.96), 4)
    0.975
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put_price(
    spot: float,
    strike: float,
    days_to_expiry: int,
    volatility: float,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Price a European put option with the Black-Scholes formula.

    >>> round(black_scholes_put_price(100, 95, 30, 0.30), 2)
    1.33
    >>> round(black_scholes_put_price(100, 100, 30, 0.30), 2)
    3.26
    >>> black_scholes_put_price(100, 95, 0, 0.30)
    Traceback (most recent call last):
        ...
    ValueError: days_to_expiry must be > 0
    >>> black_scholes_put_price(100, 95, 30, 0)
    Traceback (most recent call last):
        ...
    ValueError: volatility must be > 0
    >>> black_scholes_put_price(-1, 95, 30, 0.3)
    Traceback (most recent call last):
        ...
    ValueError: spot and strike must be > 0
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be > 0")
    if days_to_expiry <= 0:
        raise ValueError("days_to_expiry must be > 0")
    if volatility <= 0:
        raise ValueError("volatility must be > 0")

    t = days_to_expiry / DAYS_PER_YEAR
    d1 = (log(spot / strike) + (risk_free_rate + volatility**2 / 2) * t) / (
        volatility * sqrt(t)
    )
    d2 = d1 - volatility * sqrt(t)
    price = strike * exp(-risk_free_rate * t) * _normal_cdf(-d2) - spot * _normal_cdf(
        -d1
    )
    return max(price, 0.0)  # guard against tiny negative floating point noise


def probability_otm(
    spot: float,
    strike: float,
    days_to_expiry: int,
    volatility: float,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Risk-neutral probability that the put expires worthless (spot > strike),
    i.e. that you keep the premium and are not assigned.

    >>> round(probability_otm(100, 95, 30, 0.30), 3)
    0.723
    """
    t = days_to_expiry / DAYS_PER_YEAR
    d2 = (log(spot / strike) + (risk_free_rate - volatility**2 / 2) * t) / (
        volatility * sqrt(t)
    )
    return _normal_cdf(d2)


def annualized_put_yield(premium: float, strike: float, days_to_expiry: int) -> float:
    """
    Annualized (simple) yield on the cash set aside to secure the put.

    >>> round(annualized_put_yield(0.70, 100, 30), 4)
    0.0852
    >>> annualized_put_yield(1.0, 100, 0)
    Traceback (most recent call last):
        ...
    ValueError: days_to_expiry must be > 0
    >>> annualized_put_yield(-1.0, 100, 30)
    Traceback (most recent call last):
        ...
    ValueError: premium must be >= 0
    """
    if days_to_expiry <= 0:
        raise ValueError("days_to_expiry must be > 0")
    if premium < 0:
        raise ValueError("premium must be >= 0")
    if strike <= 0:
        raise ValueError("strike must be > 0")
    return premium / strike * DAYS_PER_YEAR / days_to_expiry


def required_premium(strike: float, days_to_expiry: int, target_yield: float) -> float:
    """
    Premium per share you must collect to reach ``target_yield`` (annualized).

    >>> round(required_premium(100, 30, 0.08), 4)
    0.6575
    >>> round(required_premium(100, 30, 0.10), 4)
    0.8219
    """
    if days_to_expiry <= 0:
        raise ValueError("days_to_expiry must be > 0")
    return strike * target_yield * days_to_expiry / DAYS_PER_YEAR


def find_strikes_for_target_yield(
    spot: float,
    days_to_expiry: int,
    volatility: float,
    min_yield: float = 0.08,
    max_yield: float = 0.10,
    risk_free_rate: float = 0.04,
    strike_step: float = 0.5,
) -> list[PutCandidate]:
    """
    List out-of-the-money put strikes whose premium yields between
    ``min_yield`` and ``max_yield`` annualized on the cash collateral.

    Candidates are sorted from the safest (lowest strike) to the most
    aggressive (highest strike).

    >>> for c in find_strikes_for_target_yield(100, 30, 0.25, strike_step=1):
    ...     print(c.strike, round(c.annualized_yield, 4), round(c.probability_otm, 3))
    94 0.0886 0.809
    >>> find_strikes_for_target_yield(100, 30, 0.25, min_yield=0.2, max_yield=0.1)
    Traceback (most recent call last):
        ...
    ValueError: min_yield must be <= max_yield
    """
    if min_yield > max_yield:
        raise ValueError("min_yield must be <= max_yield")
    if strike_step <= 0:
        raise ValueError("strike_step must be > 0")

    candidates = []
    strike = strike_step
    while strike < spot:  # only out-of-the-money puts
        premium = black_scholes_put_price(
            spot, strike, days_to_expiry, volatility, risk_free_rate
        )
        put_yield = annualized_put_yield(premium, strike, days_to_expiry)
        if min_yield <= put_yield <= max_yield:
            candidates.append(
                PutCandidate(
                    strike=strike,
                    premium=premium,
                    annualized_yield=put_yield,
                    probability_otm=probability_otm(
                        spot, strike, days_to_expiry, volatility, risk_free_rate
                    ),
                    breakeven=strike - premium,
                )
            )
        strike = round(strike + strike_step, 10)
    return candidates


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    spot_price, days, implied_vol = 100.0, 30, 0.30
    print(
        f"Spot ${spot_price:.2f}, {days} days to expiry, "
        f"implied volatility {implied_vol:.0%}"
    )
    print(
        f"Need ${required_premium(spot_price, days, 0.08):.2f}-"
        f"${required_premium(spot_price, days, 0.10):.2f} premium per share "
        f"on a ${spot_price:.0f} strike for 8-10% a year\n"
    )
    print(f"{'Strike':>8}{'Premium':>9}{'Yield':>8}{'P(keep)':>9}{'Breakeven':>11}")
    for c in find_strikes_for_target_yield(spot_price, days, implied_vol):
        print(
            f"{c.strike:>8.2f}{c.premium:>9.2f}{c.annualized_yield:>8.2%}"
            f"{c.probability_otm:>9.1%}{c.breakeven:>11.2f}"
        )
