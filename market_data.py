"""Yahoo adjusted daily closes, with an inclusive end date in the app."""

import pandas as pd
import yfinance as yf

from currencies import Currencies, FX_TICKER
from portfolio import MissingQuotes, PortfolioError, QuoteError, backtest, quoted_symbols


def load_prices(tickers: tuple[str, ...], start, end) -> pd.DataFrame:
    try:
        data = yf.download(
            list(tickers), start=pd.Timestamp(start).date().isoformat(),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, group_by="column",
            ignore_tz=True, progress=False, timeout=15,
        )
    except Exception as exc:
        raise MissingQuotes(tickers, f"Yahoo Finance could not return prices for {', '.join(tickers)}.") from exc
    if data is None or data.empty:
        raise MissingQuotes(tickers, f"Yahoo Finance returned no prices for {', '.join(tickers)}.")
    if "Close" not in data:
        raise MissingQuotes(tickers, "The price response is missing the adjusted Close field.")
    close = data["Close"]
    if isinstance(close, pd.Series):
        if len(tickers) != 1:
            raise MissingQuotes(tickers, "Could not match downloaded prices to their tickers.")
        close = close.to_frame(tickers[0])
    close = close.rename(columns=lambda t: str(t).upper()).reindex(columns=tickers)
    if close.isna().all().all():
        raise MissingQuotes(tickers, "No usable adjusted closing prices were returned.")
    return close


def analyze_portfolio(trades, benchmark, start, end, cash_rate, *, cached_download=load_prices, clear_cache=None,
                      initial_cash=100.0, currencies=Currencies()):
    result, _, _ = analyze_scenarios(trades, None, benchmark, start, end, cash_rate, initial_cash, currencies,
                                     cached_download=cached_download, clear_cache=clear_cache)
    return result


def analyze_scenarios(trades, scenario, benchmark, start, end, cash_rate, initial_cash, currencies, *,
                      cached_download=load_prices, clear_cache=None):
    """Replay both versions against one price snapshot. A bad alternative leaves the original available."""
    if not trades:
        raise PortfolioError("Add a transaction to the original log.")
    tickers = set(quoted_symbols(trades, benchmark, end))
    need_fx = currencies.needs_fx(tickers)
    scenario_error = None
    if scenario is not None:
        extra = set(quoted_symbols(scenario, benchmark, end))
        try:
            extra_fx = currencies.needs_fx(extra)
        except PortfolioError as exc:
            scenario, scenario_error = None, str(exc)
        else:
            tickers.update(extra)
            need_fx = need_fx or extra_fx
    if need_fx:
        tickers.add(FX_TICKER)
    tickers = tuple(sorted(tickers))
    funding = trades[0].date
    fetch_from = (funding - pd.Timedelta(days=30)).date()
    settings = dict(initial_cash=initial_cash, currencies=currencies, funding_date=funding)
    prices = pd.DataFrame()
    retried = set()
    while True:
        primary = None
        try:
            if not retried:
                prices = cached_download(tickers, fetch_from, end)
            primary = backtest(prices, trades, benchmark, start, end, cash_rate, **settings)
            alternative = None
            if scenario is not None:
                alternative = backtest(prices, scenario, benchmark, start, end, cash_rate, **settings)
            return primary, alternative, scenario_error
        except MissingQuotes as exc:
            if not retried and clear_cache is not None:
                clear_cache(tickers, fetch_from, end)
            missing = tuple(t for t in exc.tickers if t not in retried)
            if missing:
                retried.update(missing)
                try:
                    fresh = load_prices(missing, fetch_from, end)
                except MissingQuotes:
                    pass  # Keep the original row and the date of the missing close.
                else:
                    prices = fresh if prices.empty else fresh.combine_first(prices)
                    continue
            message = f"{exc} Prices for {', '.join(exc.tickers)} are still unavailable after one automatic retry."
            if primary is not None:
                return primary, None, message
            raise QuoteError(message) from exc
        except PortfolioError as exc:
            if primary is not None:
                return primary, None, str(exc)
            if isinstance(exc, QuoteError) and not retried and clear_cache is not None:
                clear_cache(tickers, fetch_from, end)
            raise
