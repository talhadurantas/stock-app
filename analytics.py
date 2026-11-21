"""Metrics derived from the same opening balance and closing valuation series."""

import numpy as np
import pandas as pd

from portfolio import SESSIONS_PER_YEAR


RISK_FREE_RATE = 0.02
PERCENT_METRICS = {
    "Total return", "CAGR", "Annual volatility", "Max drawdown", "Win rate",
    "Best day", "Worst day", "5% return quantile", "1% return quantile",
    "Expected shortfall (5%)", "Expected shortfall (1%)",
}


def drawdown(values):
    declines = values / values.cummax() - 1
    trough = declines.idxmin()
    peak = values.loc[:trough].idxmax()
    return declines, peak, trough


def stats(values):
    returns = values.pct_change(fill_method=None).dropna()
    total = values.iloc[-1] / values.iloc[0] - 1
    days = (values.index[-1] - values.index[0]).days
    with np.errstate(over="ignore"):
        cagr = np.expm1(np.log1p(total) * 365.25 / days) if days else np.nan
    cagr = cagr if np.isfinite(cagr) else np.nan
    declines, _, _ = drawdown(values)
    max_dd = declines.min()
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / SESSIONS_PER_YEAR) - 1
    excess = returns - rf_daily
    vol = returns.std() * np.sqrt(SESSIONS_PER_YEAR)
    annual_excess = excess.mean() * SESSIONS_PER_YEAR
    downside = np.sqrt(excess.clip(upper=0).pow(2).mean() * SESSIONS_PER_YEAR)
    q05, q01 = returns.quantile([0.05, 0.01]).values
    return pd.Series({
        "Total return": total, "CAGR": cagr, "Annual volatility": vol,
        "Max drawdown": max_dd,
        "Sharpe": annual_excess / vol if vol > 0 else np.nan,
        "Sortino": annual_excess / downside if downside > 0 else np.nan,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Win rate": returns.gt(0).mean(), "Best day": returns.max(), "Worst day": returns.min(),
        "5% return quantile": q05, "Expected shortfall (5%)": returns[returns <= q05].mean(),
        "1% return quantile": q01, "Expected shortfall (1%)": returns[returns <= q01].mean(),
    })


def monthly_returns(values):
    daily = values.pct_change(fill_method=None)
    # The partial opening and closing months include only observed close-to-close returns.
    return (1 + daily).resample("ME").prod(min_count=1).dropna() - 1


def rolling_metrics(values):
    n = SESSIONS_PER_YEAR
    daily = values.pct_change(fill_method=None)
    vol = daily.rolling(n).std() * np.sqrt(n)
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / n) - 1
    sharpe = (daily - rf_daily).rolling(n).mean() * n / vol.where(vol > 0)
    return pd.DataFrame({
        "Return": values.pct_change(n, fill_method=None),
        "Volatility": vol, "Sharpe": sharpe,
    })
