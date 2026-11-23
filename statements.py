"""Replay recorded fills against unadjusted marks and independently supplied balances."""

from dataclasses import dataclass
from math import fsum, isclose, isfinite

import pandas as pd

from currencies import SUPPORTED
from transactions import PortfolioError, csv_rows, parse_date, symbol


FILL_COLUMNS = "Timestamp Action Ticker Quantity Price Currency FX Fee Amount".split()
CLOSE_COLUMNS = "Date Ticker Close Currency FX".split()
BALANCE_COLUMNS = "Date Type Ticker Expected".split()
ACTIONS = {"Buy", "Sell", "Deposit", "Withdrawal", "Fee", "Dividend", "Split"}


def number(row, field, *, positive=False):
    try:
        value = float(row[field])
    except (ValueError, TypeError) as exc:
        raise PortfolioError(f"{field} needs a number.") from exc
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise PortfolioError(f"Check the value in {field}.")
    return value


def conversion(row, account):
    currency = row["Currency"].upper()
    fx = number(row, "FX", positive=True)
    if currency not in SUPPORTED or (currency == account and fx != 1):
        raise PortfolioError("Use USD or TRY and enter FX as account currency per quote currency (1 for the same currency).")
    return currency, fx


@dataclass(frozen=True)
class Fill:
    line: int
    stamp: pd.Timestamp
    action: str
    ticker: str = ""
    quantity: float = 0
    price: float = 0
    currency: str = ""
    fx: float = 1
    fee: float = 0
    amount: float = 0

    @property
    def day(self):
        return self.stamp.tz_localize(None).normalize()


def read_fills(data, account):
    if account not in SUPPORTED:
        raise PortfolioError("Choose USD or TRY as the statement account currency.")
    fills = []
    for line, row in csv_rows(data, FILL_COLUMNS):
        try:
            stamp = pd.Timestamp(row["Timestamp"])
            if pd.isna(stamp) or stamp.tz is None:
                raise PortfolioError("Include a UTC offset in the fill timestamp.")
            action = row["Action"].title()
            if action not in ACTIONS:
                raise PortfolioError("Unknown statement action.")
            fields = {}
            allowed = {"Timestamp", "Action"}
            if action in ("Buy", "Sell", "Split"):
                allowed.update(("Ticker", "Quantity"))
                fields["ticker"] = symbol(row["Ticker"])
                fields["quantity"] = number(row, "Quantity", positive=True)
                if not fields["ticker"]:
                    raise PortfolioError("Enter a ticker.")
            if action in ("Buy", "Sell"):
                allowed.update(("Price", "Currency", "FX", "Fee"))
                fields["currency"], fields["fx"] = conversion(row, account)
                fields["price"] = number(row, "Price", positive=True)
                fields["fee"] = number(row, "Fee") if row["Fee"] else 0.0
            elif action != "Split":
                allowed.add("Amount")
                fields["amount"] = number(row, "Amount", positive=True)
            unexpected = [k for k, v in row.items() if v and k not in allowed]
            if unexpected:
                raise PortfolioError(f"Leave {', '.join(unexpected)} blank for {action}.")
            fills.append(Fill(line, stamp.tz_convert("UTC"), action, **fields))
        except (ValueError, OverflowError) as exc:
            raise PortfolioError(f"Activity CSV line {line}: {exc}") from exc
    if not fills:
        raise PortfolioError("The statement activity CSV is empty.")
    return sorted(fills, key=lambda f: f.stamp)


def read_closes(data, fills, account):
    quotes = {}
    currencies = {}
    for fill in fills:
        if fill.currency and currencies.setdefault(fill.ticker, fill.currency) != fill.currency:
            raise PortfolioError(f"Inconsistent fill currency for {fill.ticker}.")
    for line, row in csv_rows(data, CLOSE_COLUMNS):
        try:
            key = parse_date(row["Date"]), symbol(row["Ticker"])
            day, ticker = key
            currency, fx = conversion(row, account)
            if not ticker or key in quotes:
                raise PortfolioError("Each date needs at most one close per ticker.")
            if currencies.setdefault(ticker, currency) != currency:
                raise PortfolioError(f"The quote currency for {ticker} changed.")
            quotes[key] = number(row, "Close", positive=True) * fx
        except PortfolioError as exc:
            raise PortfolioError(f"Closes CSV line {line}: {exc}") from exc
    return pd.DataFrame([(d, t, p) for (d, t), p in quotes.items()],
                        columns=["Date", "Ticker", "Price"]).pivot(index="Date", columns="Ticker", values="Price").sort_index()


def read_balances(data):
    dates, expected = set(), {}
    for line, row in csv_rows(data, BALANCE_COLUMNS):
        try:
            dates.add(parse_date(row["Date"]))
            kind, ticker = row["Type"].title(), symbol(row["Ticker"])
            if (kind == "Cash" and ticker) or (kind == "Holding" and not ticker) or kind not in ("Cash", "Holding"):
                raise PortfolioError("Use Cash with a blank ticker or Holding with a ticker.")
            if (kind, ticker) in expected:
                raise PortfolioError("This balance appears twice.")
            expected[kind, ticker] = number(row, "Expected")
        except PortfolioError as exc:
            raise PortfolioError(f"Balances CSV line {line}: {exc}") from exc
    if len(dates) != 1 or ("Cash", "") not in expected:
        raise PortfolioError("Supply closing cash and holdings for one statement date.")
    return dates.pop(), expected


@dataclass
class StatementResult:
    daily: pd.DataFrame
    holdings: dict
    cash: float


def replay_statement(fills, closes, opening_cash, end, *, skipped=()):
    if not isfinite(opening_cash) or opening_cash < 0:
        raise PortfolioError("Opening cash must be finite and nonnegative.")
    start = fills[0].day
    if end < start:
        raise PortfolioError("The balance date precedes the first activity.")
    activities = {}
    for fill in fills:
        if fill.day <= end:
            activities.setdefault(fill.day, []).append(fill)
    days = sorted(set(closes.loc[start:end].index) | activities.keys() | {end})
    cash, fund_units, nav = float(opening_cash), opening_cash / 100, 100.0
    holdings, marks, records = {}, {}, []
    for day in days:
        traded = set()
        flow = fees = 0.0
        for fill in activities.get(day, []):
            if fill.line in skipped:
                continue
            ticker, amount, qty = fill.ticker, fill.amount, fill.quantity
            if fill.action in ("Deposit", "Withdrawal"):
                balance = cash + fsum(q * marks[t] for t, q in holdings.items())
                if fund_units > 0:
                    nav = balance / fund_units
                delta = amount if fill.action == "Deposit" else -amount
                if nav <= 0:
                    raise PortfolioError(f"Activity line {fill.line}: cannot unitize a cash flow at zero equity.")
                fund_units = max(0.0, fund_units + delta / nav)
                cash += delta
                flow += delta
            elif fill.action in ("Buy", "Sell"):
                change = qty if fill.action == "Buy" else -qty
                marks[ticker] = fill.price * fill.fx
                cash -= change * marks[ticker] + fill.fee
                fees += fill.fee
                holdings[ticker] = holdings.get(ticker, 0.0) + change
                traded.add(ticker)
            elif fill.action == "Split":
                if ticker in holdings:
                    holdings[ticker] *= qty
                    marks[ticker] /= qty
                    traded.add(ticker)
            elif fill.action == "Dividend":
                cash += amount
            else:
                cash -= amount
                fees += amount
            if cash < -1e-8 or (ticker and holdings.get(ticker, 0) < -1e-10):
                raise PortfolioError(f"Activity line {fill.line} ({fill.action} {ticker}): insufficient cash or shares.")
            if abs(cash) < 1e-9:
                cash = 0.0
            if ticker in holdings and abs(holdings[ticker]) < 1e-10:
                del holdings[ticker]
        observed = closes.loc[day].dropna().to_dict() if day in closes.index else {}
        missing = traded.intersection(holdings).difference(observed)
        if missing:
            raise PortfolioError(f"Missing unadjusted close for {', '.join(sorted(missing))} on {day.date()}.")
        marks.update(observed)
        balance = cash + fsum(q * marks[t] for t, q in holdings.items())
        if fund_units > 0:
            nav = balance / fund_units
        records.append((day, balance, cash, flow, fees, nav))
    daily = pd.DataFrame(records, columns=["Date", "Balance", "Cash", "External flow", "Fees", "Performance per 100"]).set_index("Date")
    return StatementResult(daily, holdings, cash)


def reconcile(result, expected):
    actual = {("Cash", ""): result.cash, **{("Holding", t): q for t, q in result.holdings.items()}}
    rows = []
    for kind, ticker in sorted(expected.keys() | actual.keys()):
        reported, replayed = expected.get((kind, ticker), 0.0), actual.get((kind, ticker), 0.0)
        tolerance = 0.005000001 if kind == "Cash" else 1e-8
        rows.append((kind, ticker, reported, replayed, replayed - reported,
                     isclose(replayed, reported, abs_tol=tolerance, rel_tol=0)))
    return pd.DataFrame(rows, columns=["Type", "Ticker", "Statement", "Replay", "Difference", "Match"])
