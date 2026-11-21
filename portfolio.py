"""Transaction ledger for a fractional, single-currency portfolio."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from math import fsum, isfinite

import numpy as np
import pandas as pd


COLUMNS = ["Date", "Action", "Sell", "Buy", "Weights"]
ACTIONS = ("Initial", "Buy", "Sell", "Sell & Buy", "Rebalance")
SESSIONS_PER_YEAR = 252


class PortfolioError(ValueError):
    """A problem the user can resolve in the log, settings or price data."""


class QuoteError(PortfolioError):
    """The requested portfolio cannot be valued using this price response."""


@dataclass(frozen=True)
class Trade:
    row: int
    date: pd.Timestamp
    action: str
    sells: tuple[str, ...]
    buys: tuple[str, ...]
    fractions: tuple[float, ...]


@dataclass
class Backtest:
    ledger: pd.DataFrame       # Closing portfolio, benchmark and cash, rebased to 100
    positions: pd.DataFrame    # Closing value by ticker, in the same units
    executions: pd.DataFrame   # Post-trade snapshots, including trades on the same close
    pending: tuple[int, ...]
    carried: dict[str, int]    # Held-position marks carried forward in the report


def symbol(value: str) -> str:
    name = value.strip().upper()
    # Yahoo's class-share spelling for the tickers in historical exports.
    return {"BRK.B": "BRK-B", "BRK.A": "BRK-A"}.get(name, name)


def cell(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def parse_date(value) -> pd.Timestamp:
    raw = cell(value)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y %m %d", "%Y-%m-%d 00:00:00"):
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            pass
    raise PortfolioError(f"Invalid date {raw!r}. Use YYYY-MM-DD or YYYYMMDD.")


def parse_log(frame: pd.DataFrame) -> list[Trade]:
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise PortfolioError("Missing transaction columns: " + ", ".join(missing))
    trades = []
    for number, values in enumerate(frame[COLUMNS].itertuples(index=False, name=None), 1):
        raw_date, action, sell, buy, weights = map(cell, values)
        if not any((raw_date, action, sell, buy, weights)):
            continue
        try:
            day = parse_date(raw_date)
            actions = {a.casefold(): a for a in ACTIONS}
            if action.casefold() not in actions:
                raise PortfolioError("Choose " + ", ".join(ACTIONS) + ".")
            action = actions[action.casefold()]
            sells = tuple(symbol(t) for t in sell.split(",")) if sell else ()
            buys = tuple(symbol(t) for t in buy.split(",")) if buy else ()
            for tickers in (sells, buys):
                if "" in tickers or len(tickers) != len(set(tickers)):
                    raise PortfolioError("Remove empty or duplicate tickers within each field.")
            if action in ("Sell", "Sell & Buy") and not sells:
                raise PortfolioError(f"{action} needs a sell ticker.")
            if action in ("Buy", "Sell & Buy") and not buys:
                raise PortfolioError(f"{action} needs a buy ticker.")
            if sells and action not in ("Sell", "Sell & Buy"):
                raise PortfolioError(f"Leave Sell blank for {action}.")
            if action == "Sell" and (buys or weights):
                raise PortfolioError("Leave Buy and Weights blank for a sale.")
            if weights:
                try:
                    percentages = tuple(float(w.strip()) for w in weights.split(","))
                except ValueError as exc:
                    raise PortfolioError("Enter numeric percentages separated by commas.") from exc
                if len(percentages) != len(buys):
                    raise PortfolioError("Enter one percentage for each buy ticker.")
                if any(not isfinite(w) or w < 0 or w > 100 for w in percentages):
                    raise PortfolioError("Each percentage must be finite and between 0 and 100.")
                if fsum(percentages) > 100 + 1e-10:
                    raise PortfolioError("Percentages add up to more than 100.")
                fractions = tuple(w / 100 for w in percentages)
            else:
                fractions = (1 / len(buys),) * len(buys) if buys else ()
        except PortfolioError as exc:
            raise PortfolioError(f"Row {number}: {exc}") from exc
        trades.append(Trade(number, day, action, sells, buys, fractions))
    if not trades:
        raise PortfolioError("Add a transaction to the log.")
    trades.sort(key=lambda t: t.date)  # Python's stable sort preserves same-date input order.
    for trade in trades[1:]:
        if trade.action == "Initial":
            raise PortfolioError(f"Row {trade.row}: Initial is only valid as the first transaction.")
    return trades


def import_log(data: bytes) -> pd.DataFrame:
    try:
        reader = csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader, [])
        names = [v.strip().casefold() for v in header]
        if len(set(names)) != len(names):
            raise PortfolioError("The CSV contains duplicate column names.")
        if any(c.casefold() not in names for c in COLUMNS):
            raise PortfolioError("The transaction CSV needs these columns: " + ", ".join(COLUMNS) + ".")
        slots = [names.index(c.casefold()) for c in COLUMNS]
        rows = []
        for fields in reader:
            if not any(field.strip() for field in fields):
                continue
            if len(fields) != len(header):
                raise PortfolioError(f"CSV line {reader.line_num}: expected {len(header)} fields. Quote fields containing commas.")
            rows.append([fields[i].strip() for i in slots])
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PortfolioError("Could not read the file as a UTF-8 transaction CSV.") from exc
    frame = pd.DataFrame(rows, columns=COLUMNS, dtype="string")
    parse_log(frame)  # Validate the entire replacement before the editor changes.
    return frame


def export_log(frame: pd.DataFrame) -> bytes:
    return frame[COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")


def quoted_symbols(trades: list[Trade], benchmark: str, end) -> tuple[str, ...]:
    last = pd.Timestamp(end)
    tickers = {symbol(benchmark)}
    for trade in trades:
        if trade.date <= last:
            tickers.update(trade.sells)
            tickers.update(t for t, fraction in zip(trade.buys, trade.fractions) if fraction)
    return tuple(sorted(tickers))


def execute(trade: Trade, day, cash, units, values, quotes):
    """Execute at today's close. Values and units describe the same positions."""
    total = cash + fsum(values.values())
    sells = tuple(units) if trade.action == "Rebalance" else trade.sells
    missing = [t for t in sells if t not in units]
    if missing:
        raise PortfolioError(f"Row {trade.row}: cannot sell an unheld position ({', '.join(missing)}).")
    sold = fsum(values[t] for t in sells)
    if trade.action == "Buy":
        budget = cash
    elif trade.action == "Sell & Buy":
        budget = sold
    else:
        budget = total
    orders = {t: budget * w for t, w in zip(trade.buys, trade.fractions) if budget * w > 0}
    needs_quote = set(sells).union(orders)
    unavailable = sorted(t for t in needs_quote if not np.isfinite(quotes.get(t, np.nan)))
    if unavailable:
        raise QuoteError(f"Row {trade.row}: no fresh adjusted close for {', '.join(unavailable)} on {day.date()}. Check the symbol, listing date or trading calendar.")
    for ticker in sells:
        del units[ticker]
        del values[ticker]
    for ticker, amount in orders.items():
        units[ticker] = units.get(ticker, 0.0) + amount / quotes[ticker]
        values[ticker] = values.get(ticker, 0.0) + amount
    spent = fsum(orders.values())
    cash = fsum((cash, sold, -spent))
    if abs(cash) <= total * 1e-12:
        cash = 0.0
    note = "No cash available" if trade.action == "Buy" and budget == 0 else ""
    snapshot = {
        "Log row": trade.row, "Requested": trade.date, "Executed": day, "Action": trade.action,
        "Sold": sold, "Bought": spent, "Cash after": cash,
        "Cash %": 100 * cash / total,
        "Holdings after": ", ".join(f"{t} {100*v/total:.2f}%" for t, v in sorted(values.items())) or "All cash",
        "Note": note,
    }
    return cash, snapshot


def backtest(prices: pd.DataFrame, trades: list[Trade], benchmark: str, start, end,
             cash_rate: float = 4.5) -> Backtest:
    """Mark yesterday's holdings first, accrue cash, then execute at the close."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    bench = symbol(benchmark)
    if start > end or not isfinite(cash_rate) or cash_rate < 0:
        raise PortfolioError("Check the date range and annual cash interest percentage.")
    if not trades or trades[0].date > end:
        raise PortfolioError("The log has no transaction on or before the end date.")
    if prices.empty or bench not in prices:
        raise QuoteError(f"No adjusted prices available for benchmark {bench}.")
    prices = prices.copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    if prices.index.has_duplicates or prices.columns.has_duplicates:
        raise QuoteError("Price data contains duplicate dates or tickers.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce").astype(float)
    prices = prices.loc[:end]
    invalid = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
    if invalid.any().any():
        raise QuoteError("Prices must be positive and finite wherever a quote is supplied.")
    benchmark_prices = prices[bench].dropna().loc[trades[0].date:end]
    calendar = benchmark_prices.index
    if calendar.empty or calendar[-1] < start:
        raise QuoteError("No benchmark sessions overlap the funded portfolio and selected dates.")
    # An ETF launched years after the log began must not collapse old trades onto its launch date.
    first_order = trades[0].date
    if calendar[0] > first_order and not prices.loc[:first_order, bench].notna().any():
        before_benchmark = prices.loc[(prices.index >= first_order) & (prices.index < calendar[0])]
        if before_benchmark.drop(columns=bench).notna().any().any():
            raise QuoteError(f"Portfolio quotes predate the available {bench} history. Choose a benchmark covering the first transaction ({first_order.date()}).")
    scheduled = {}
    pending = []
    for trade in trades:
        idx = calendar.searchsorted(trade.date)
        if idx == len(calendar):
            pending.append(trade.row)
        else:
            scheduled.setdefault(calendar[idx], []).append(trade)
    tickers = sorted({t for group in scheduled.values() for trade in group for t in (*trade.sells, *trade.buys)})
    asset_prices = prices.reindex(columns=tickers)
    # A past quote may value a held asset on a different market holiday.
    # The raw series is used separately for every actual execution.
    raw = asset_prices.reindex(calendar)
    marks = asset_prices.ffill().reindex(calendar)
    raw_rows = raw.itertuples(index=False, name=None) if tickers else (() for _ in calendar)
    mark_rows = marks.itertuples(index=False, name=None) if tickers else (() for _ in calendar)
    cash, units = 100.0, {}
    cash_factor = (1 + cash_rate / 100) ** (1 / SESSIONS_PER_YEAR)
    daily, positions, snapshots, carried = [], [], [], {}
    for i, (day, observed, marked) in enumerate(zip(calendar, raw_rows, mark_rows)):
        quotes = dict(zip(tickers, observed))
        marks_today = dict(zip(tickers, marked))
        if i:
            cash *= cash_factor
        values = {ticker: quantity * marks_today[ticker] for ticker, quantity in units.items()}
        if day >= start:
            for ticker in units:
                if not np.isfinite(quotes[ticker]):
                    carried[ticker] = carried.get(ticker, 0) + 1
        for trade in scheduled.get(day, ()):
            cash, snapshot = execute(trade, day, cash, units, values, quotes)
            if day >= start:
                snapshots.append(snapshot)
        if day >= start:
            daily.append((day, cash + fsum(values.values()), cash))
            positions.append(values.copy())

    ledger = pd.DataFrame(daily, columns=["Date", "Portfolio", "Cash"]).set_index("Date")
    scale = 100 / ledger["Portfolio"].iloc[0]
    ledger *= scale
    benchmark_window = benchmark_prices.reindex(ledger.index)
    ledger["Benchmark"] = 100 * (benchmark_window / benchmark_window.iloc[0])
    assets = pd.DataFrame(positions, index=ledger.index, columns=tickers, dtype=float).fillna(0) * scale
    events = pd.DataFrame(snapshots, columns=["Log row", "Requested", "Executed", "Action", "Sold", "Bought", "Cash after", "Cash %", "Holdings after", "Note"])
    if not events.empty:
        events[["Sold", "Bought", "Cash after"]] *= scale
    return Backtest(ledger[["Portfolio", "Benchmark", "Cash"]], assets, events, tuple(pending), carried)
