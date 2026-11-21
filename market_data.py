"""Yahoo adjusted daily closes, with an inclusive end date in the app."""

import pandas as pd
import yfinance as yf

from portfolio import PortfolioError


def load_prices(tickers: tuple[str, ...], start, end) -> pd.DataFrame:
    try:
        data = yf.download(
            list(tickers), start=pd.Timestamp(start).date().isoformat(),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, group_by="column",
            ignore_tz=True, progress=False, timeout=15,
        )
    except Exception as exc:
        raise PortfolioError(f"Yahoo Finance could not return prices. {exc}") from exc
    if data is None or data.empty:
        raise PortfolioError("Yahoo Finance returned no prices. Check the tickers and dates, then retry.")
    if "Close" not in data:
        raise PortfolioError("The price response is missing the adjusted Close field.")
    close = data["Close"]
    if isinstance(close, pd.Series):
        if len(tickers) != 1:
            raise PortfolioError("Could not match downloaded prices to their tickers.")
        close = close.to_frame(tickers[0])
    close = close.rename(columns=lambda t: str(t).upper()).reindex(columns=tickers)
    if close.isna().all().all():
        raise PortfolioError("No usable adjusted closing prices were returned. Check the tickers or retry.")
    return close
