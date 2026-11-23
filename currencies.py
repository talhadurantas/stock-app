"""Quote, account and reporting currencies for the portfolio replay."""

from dataclasses import dataclass

import pandas as pd

from transactions import MissingQuotes, PortfolioError, symbol


FX_TICKER = "TRY=X"  # Turkish lira per US dollar
NATIVE = "As quoted"
SUPPORTED = ("USD", "TRY")


@dataclass(frozen=True)
class Currencies:
    account: str = "USD"
    report: str = "USD"
    overrides: tuple = ()

    def quote(self, ticker):
        if self.account == NATIVE:
            return NATIVE
        override = dict(self.overrides).get(ticker)
        if override:
            return override
        if ticker.endswith(".IS") or ticker == FX_TICKER:
            return "TRY"
        if "." in ticker or ticker.endswith("=X"):
            raise PortfolioError(f"Declare the USD or TRY quote currency for {ticker}, or use As quoted for a single-currency log.")
        return "USD"

    def needs_fx(self, tickers):
        quotes = {self.quote(t) for t in tickers}
        return self.account != self.report or bool(quotes - {self.account})


def currency_settings(account, report, overrides=""):
    if account == NATIVE:
        return Currencies(NATIVE, NATIVE)
    if account not in SUPPORTED or report not in SUPPORTED:
        raise PortfolioError("Choose USD or TRY for both account and report.")
    known = {}
    for entry in overrides.split(",") if overrides.strip() else ():
        ticker, sep, currency = entry.strip().rpartition("=")
        ticker, currency = symbol(ticker), currency.strip().upper()
        if not sep or not ticker or currency not in SUPPORTED or ticker in known:
            raise PortfolioError("Use one override per symbol, such as ABC=TRY, XYZ=USD.")
        known[ticker] = currency
    return Currencies(account, report, tuple(sorted(known.items())))


def exchange_factor(source, target, rates):
    if source == target:
        return 1.0
    return rates if source == "USD" else 1 / rates


def valuation_prices(prices, tickers, bench, calendar, currencies):
    """Carry native holding marks and FX independently. Orders require native execution quotes."""
    source = prices.reindex(columns=tickers)
    raw = source.reindex(calendar).copy()
    marks = source.ffill().reindex(calendar).copy()
    rate = pd.Series(1.0, index=calendar)
    if currencies.needs_fx([*tickers, bench]):
        rate = prices.reindex(columns=[FX_TICKER])[FX_TICKER].ffill().reindex(calendar)
        if rate.isna().any():
            day = rate.index[rate.isna()][0].date()
            raise MissingQuotes((FX_TICKER,), f"No USD/TRY rate on or before {day}.")
    for ticker in tickers:
        factor = exchange_factor(currencies.quote(ticker), currencies.account, rate)
        raw[ticker] *= factor
        marks[ticker] *= factor
    report_fx = pd.Series(exchange_factor(currencies.account, currencies.report, rate), index=calendar)
    benchmark = prices.loc[calendar, bench] * exchange_factor(currencies.quote(bench), currencies.report, rate)
    return raw, marks, report_fx, benchmark
