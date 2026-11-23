"""Exchange sessions and close times used to schedule modeled orders."""

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from transactions import PortfolioError, QuoteError


def venue(ticker):
    if ticker.endswith(".IS"):
        return "XIST"
    if "." not in ticker and not ticker.endswith("=X"):
        return "XNYS"
    # An override may specify a currency without identifying an exchange.
    return ticker


@lru_cache(maxsize=24)
def exchange_closes(exchange, first_year, last_year):
    try:
        calendar = xcals.get_calendar(exchange, start=f"{first_year}-01-01", end=f"{last_year}-12-31")
    except ValueError as exc:
        raise PortfolioError(f"The {exchange} calendar does not cover {first_year}–{last_year}. {exc}") from exc
    return calendar.schedule["close"].dt.tz_convert(None)


class Sessions:
    def __init__(self, prices, tickers, account_currency, start, end):
        markets = {venue(t) for t in tickers}
        self.cash_market = "XIST" if account_currency == "TRY" or (
            account_currency == "As quoted" and markets == {"XIST"}) else "XNYS"
        self.closes = {}
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for market in markets | {self.cash_market}:
            if market in ("XIST", "XNYS"):
                times = exchange_closes(market, start.year, end.year).loc[start:end]
            else:
                # Without exchange metadata the observed sessions are the supported history.
                days = prices.reindex(columns=[market])[market].dropna().loc[start:end].index
                times = pd.Series(days + pd.Timedelta(hours=23, minutes=59), index=days)
            self.closes[market] = pd.DatetimeIndex(times[times <= now])
        combined = pd.DatetimeIndex([])
        for times in self.closes.values():
            combined = combined.union(times.normalize())
        self.days = combined.sort_values()
        self.cash_days = self.closes[self.cash_market].normalize()
        if self.days.empty:
            raise QuoteError("No portfolio exchange sessions overlap the funded period.")

    def next_close(self, market, earliest):
        times = self.closes[market]
        if times.empty and market not in ("XIST", "XNYS"):
            # Route an entirely absent symbol to the normal missing-quote retry.
            times = self.closes[self.cash_market]
        i = times.searchsorted(earliest)
        return times[i] if i < len(times) else None
