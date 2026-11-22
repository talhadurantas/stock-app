"""Yahoo adjusted daily closes, with an inclusive end date in the app."""

import pandas as pd
import yfinance as yf

from portfolio import MissingQuotes, QuoteError, backtest, quoted_symbols


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


def analyze_portfolio(trades, benchmark, start, end, cash_rate, *, cached_download=load_prices, clear_cache=None):
    tickers = quoted_symbols(trades, benchmark, end)
    # A prior close distinguishes an opening holiday from a younger benchmark.
    fetch_from = (trades[0].date - pd.Timedelta(days=30)).date()
    prices = pd.DataFrame()
    retried = set()
    while True:
        try:
            if not retried:
                prices = cached_download(tickers, fetch_from, end)
            return backtest(prices, trades, benchmark, start, end, cash_rate)
        except MissingQuotes as exc:
            if not retried and clear_cache is not None:
                clear_cache(tickers, fetch_from, end)
            missing = tuple(t for t in exc.tickers if t not in retried)
            if missing:
                retried.update(missing)
                try:
                    fresh = load_prices(missing, fetch_from, end)
                except MissingQuotes:
                    pass  # Report the original missing trade close, including its row and date.
                else:
                    prices = fresh if prices.empty else fresh.combine_first(prices)
                    continue
            names = ", ".join(exc.tickers)
            raise QuoteError(f"{exc} Prices for {names} are still unavailable after one automatic retry.") from exc
        except QuoteError:
            if not retried and clear_cache is not None:
                clear_cache(tickers, fetch_from, end)
            raise
