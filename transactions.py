"""Transaction formats, validation and Yahoo ticker names."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from math import fsum, isfinite

import pandas as pd


COLUMNS = ["Date", "Action", "Sell", "Buy", "Weights"]
SELL_PERCENT = "Sell %"
OPTIONAL_COLUMNS = [SELL_PERCENT, "Amount", "Fee"]
LOG_COLUMNS = [*COLUMNS, *OPTIONAL_COLUMNS]
BUY_AND_HOLD = "Buy and hold"
CASH_ACTIONS = ("Deposit", "Withdrawal", "Fee")
ACTIONS = ("Initial", "Buy", "Sell", "Sell & Buy", "Rebalance", *CASH_ACTIONS)
SESSIONS_PER_YEAR = 252


class PortfolioError(ValueError):
    """A problem the user can resolve in the log, settings or price data."""


class QuoteError(PortfolioError):
    """The requested portfolio cannot be valued using this price response."""


class MissingQuotes(QuoteError):
    """A fresh download may supply the prices needed for these symbols."""

    def __init__(self, tickers, message):
        super().__init__(message)
        self.tickers = tuple(tickers)


@dataclass(frozen=True)
class Trade:
    row: int
    date: pd.Timestamp
    action: str
    sells: tuple[str, ...]
    buys: tuple[str, ...]
    fractions: tuple[float, ...]
    sell_fractions: tuple[float, ...] = ()   # Blank on legacy rows means sell all.
    amount: float = 0.0
    fee: float = 0.0


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


def percentages(text: str, tickers: tuple[str, ...], label: str) -> tuple[float, ...]:
    parts = text.split(",")
    if len(parts) != len(tickers):
        raise PortfolioError(f"{label} needs one percentage per ticker, in the same order.")
    try:
        numbers = [float(part.strip().removesuffix("%")) for part in parts]
    except ValueError as exc:
        raise PortfolioError(f"{label} needs numeric percentages separated by commas.") from exc
    if any(not isfinite(n) or n < 0 or n > 100 for n in numbers):
        raise PortfolioError(f"Each percentage in {label} must be finite and between 0 and 100.")
    return tuple(n / 100 for n in numbers)


def money(text, label):
    try:
        value = float(text) if text else 0.0
    except ValueError as exc:
        raise PortfolioError(f"{label} must be a number.") from exc
    if not isfinite(value) or value < 0:
        raise PortfolioError(f"{label} must be finite and nonnegative.")
    return value


def parse_log(frame: pd.DataFrame, *, skip_rows=(), allow_empty=False) -> list[Trade]:
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise PortfolioError("Missing transaction columns: " + ", ".join(missing))
    trades = []
    full_log = frame.reindex(columns=LOG_COLUMNS, fill_value="")
    for number, values in enumerate(full_log.itertuples(index=False, name=None), 1):
        fields = list(map(cell, values))
        if number in skip_rows or not any(fields):
            continue
        raw_date, action, sell, buy, weights, sell_pct, raw_amount, raw_fee = fields
        try:
            day = parse_date(raw_date)
            actions = {a.casefold(): a for a in ACTIONS}
            actions["withdraw"] = "Withdrawal"
            if action.casefold() not in actions:
                raise PortfolioError("Choose " + ", ".join(ACTIONS) + ".")
            action = actions[action.casefold()]
            amount, fee = money(raw_amount, "Amount"), money(raw_fee, "Fee")
            if action in CASH_ACTIONS:
                if amount <= 0 or any((sell, buy, weights, sell_pct)):
                    raise PortfolioError(f"{action} needs a positive Amount and blank ticker and percentage fields.")
                if action == "Fee" and fee:
                    raise PortfolioError("Enter a standalone fee in Amount and leave Fee blank.")
            elif amount:
                raise PortfolioError("Amount is for Deposit, Withdrawal and Fee rows.")
            sells = tuple(symbol(t) for t in sell.split(",")) if sell else ()
            buys = tuple(symbol(t) for t in buy.split(",")) if buy else ()
            for tickers in (sells, buys):
                if "" in tickers or len(tickers) != len(set(tickers)):
                    raise PortfolioError("Remove empty or duplicate tickers within each field.")
            if action in ("Sell", "Sell & Buy") and not sells:
                raise PortfolioError(f"{action} needs a sell ticker.")
            if action in ("Buy", "Sell & Buy") and not buys:
                raise PortfolioError(f"{action} needs a buy ticker.")
            if (sells or sell_pct) and action not in ("Sell", "Sell & Buy"):
                raise PortfolioError(f"Leave Sell and Sell % blank for {action}.")
            if action == "Sell" and (buys or weights):
                raise PortfolioError("Leave Buy and Weights blank for a sale.")
            if weights:
                fractions = percentages(weights, buys, "Buy weights")
                if fsum(fractions) > 1 + 1e-12:
                    raise PortfolioError("Percentages add up to more than 100.")
            else:
                fractions = (1 / len(buys),) * len(buys) if buys else ()
            sell_fractions = percentages(sell_pct, sells, SELL_PERCENT) if sell_pct else ()
        except PortfolioError as exc:
            raise PortfolioError(f"Row {number}: {exc}") from exc
        trades.append(Trade(number, day, action, sells, buys, fractions, sell_fractions, amount, fee))
    if not trades and not allow_empty:
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
        slots = [names.index(c.casefold()) if c.casefold() in names else None for c in LOG_COLUMNS]
        rows = []
        for fields in reader:
            if not any(field.strip() for field in fields):
                continue
            if len(fields) != len(header):
                raise PortfolioError(f"CSV line {reader.line_num}: expected {len(header)} fields. Quote fields containing commas.")
            rows.append([fields[i].strip() if i is not None else "" for i in slots])
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PortfolioError("Could not read the file as a UTF-8 transaction CSV.") from exc
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS, dtype="string")
    parse_log(frame)  # Validate the entire replacement before the editor changes.
    return frame


def export_log(frame: pd.DataFrame) -> bytes:
    columns = COLUMNS + [c for c in OPTIONAL_COLUMNS if c in frame and frame[c].map(cell).ne("").any()]
    return frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")


def quoted_symbols(trades: list[Trade], benchmark: str, end) -> tuple[str, ...]:
    last = pd.Timestamp(end)
    tickers = {symbol(benchmark)}
    for trade in trades:
        if trade.date <= last:
            tickers.update(trade.sells)
            tickers.update(t for t, fraction in zip(trade.buys, trade.fractions) if fraction)
    return tuple(sorted(tickers))
