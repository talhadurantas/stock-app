"""Close-by-close replay, account valuation and holding contributions."""

from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
import pandas as pd

from currencies import Currencies, FX_TICKER, valuation_prices
from orders import OrderBook, execute
from sessions import Sessions
from transactions import (
    ACTIONS, BUY_AND_HOLD, CASH_ACTIONS, COLUMNS, LOG_COLUMNS, SELL_PERCENT, SESSIONS_PER_YEAR,
    MissingQuotes, PortfolioError, QuoteError, Trade, export_log, import_log, parse_log, quoted_symbols, symbol,
)


@dataclass
class Backtest:
    ledger: pd.DataFrame       # Strategy values and the active portfolio's cash
    positions: pd.DataFrame    # Closing value by ticker, in the same units
    executions: pd.DataFrame   # Post-trade snapshots, including trades on the same close
    pending: tuple[int, ...]
    carried: dict[str, int]    # Held-position marks carried forward in the report
    hold_carried: dict[str, int]
    account: pd.DataFrame       # Actual money in the reporting currency
    contributions: pd.DataFrame # Daily additive index points by position, cash and fees
    currency: str
    no_orders: pd.DataFrame    # Includes earlier instructions used to form the opening book
    benchmark_closes: pd.DatetimeIndex  # Native observations, excluding carried marks


def backtest(prices: pd.DataFrame, trades: list[Trade], benchmark: str, start, end,
             cash_rate: float = 4.5, *, initial_cash=100.0, currencies=Currencies(), funding_date=None, valuation_symbols=None) -> Backtest:
    """Mark yesterday's holdings first, accrue cash, then execute at the close."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    bench = symbol(benchmark)
    if start > end or not isfinite(cash_rate) or cash_rate < 0:
        raise PortfolioError("Check the date range and annual cash interest percentage.")
    if not isfinite(initial_cash) or initial_cash < 0:
        raise PortfolioError("Starting cash must be finite and nonnegative.")
    if funding_date is None and not trades:
        raise PortfolioError("The log has no transaction on or before the end date.")
    first_order = pd.Timestamp(funding_date if funding_date is not None else trades[0].date)
    if trades and trades[0].date < first_order:
        raise PortfolioError(f"Row {trades[0].row}: the scenario predates the shared funding date {first_order.date()}.")
    if first_order > end:
        raise PortfolioError("The log has no transaction on or before the end date.")
    if prices.empty or bench not in prices:
        raise MissingQuotes((bench,), f"No adjusted prices available for benchmark {bench}.")
    prices = prices.copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    if prices.index.has_duplicates or prices.columns.has_duplicates:
        raise QuoteError("Price data contains duplicate dates or tickers.")
    tickers = sorted({t for trade in trades if trade.date <= end for t in (*trade.sells, *trade.buys)})
    shared = set(tickers if valuation_symbols is None else valuation_symbols)
    used = shared | {bench}
    if currencies.needs_fx([*tickers, bench]):
        used.add(FX_TICKER)
    prices = prices.sort_index().loc[:end, prices.columns.isin(used)].apply(pd.to_numeric, errors="coerce").astype(float)
    invalid = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
    if invalid.any().any():
        raise QuoteError("Prices must be positive and finite wherever a quote is supplied.")
    observed = prices.reindex(columns=sorted(shared | {bench})).dropna(how="all")
    if observed.empty:
        raise MissingQuotes((bench,), f"No {bench} prices overlap the funded portfolio and selected dates.")
    sessions = Sessions(prices, shared, currencies.account, first_order, end)
    calendar = sessions.days
    if calendar[-1] < start:
        raise QuoteError("No portfolio exchange sessions fall in the selected reporting window.")
    benchmark_history = prices[bench].dropna()
    if benchmark_history.empty:
        raise MissingQuotes((bench,), f"No adjusted prices available for benchmark {bench}.")
    if benchmark_history.index[0] > calendar[0]:
        raise QuoteError(f"Portfolio sessions predate the available {bench} history. Choose a benchmark covering the funding date.")
    book = OrderBook(trades, sessions)
    raw, marks, report_fx, benchmark_prices, exchange_rates = valuation_prices(prices, tickers, bench, calendar, currencies)
    raw_rows = raw.itertuples(index=False, name=None) if tickers else (() for _ in calendar)
    mark_rows = marks.itertuples(index=False, name=None) if tickers else (() for _ in calendar)
    cash, units = float(initial_cash), {}
    initial_fx = report_fx.iloc[0]
    shares = cash * initial_fx / 100.0  # Fund units. Deposits and withdrawals issue/redeem at the current value.
    unit_price = 100.0
    hold_cash, original_units = 100.0 / initial_fx, {}
    cash_factor = (1 + cash_rate / 100) ** (1 / SESSIONS_PER_YEAR)
    net_invested = cash * initial_fx
    prior_values, prior_cash = {}, net_invested
    daily, positions, accounts, contributions, snapshots = [], [], [], [], []
    carried, hold_carried = {}, {}
    first_cash_day = sessions.cash_days[0] if len(sessions.cash_days) else None
    for i, (day, observed, marked, fx) in enumerate(zip(calendar, raw_rows, mark_rows, report_fx)):
        quotes = dict(zip(tickers, observed))
        marks_today = dict(zip(tickers, marked))
        if first_cash_day is not None and day > first_cash_day and day in sessions.cash_days:
            cash *= cash_factor
            hold_cash *= cash_factor
        values = {t: q * marks_today[t] for t, q in units.items()}
        attribution = {"Cash": 0.0, "Fees": 0.0}
        if shares:
            attribution.update({t: (v * fx - prior_values[t]) / shares for t, v in values.items()})
            attribution["Cash"] = (cash * fx - prior_cash) / shares
        if day >= start:
            for owned, counts in ((units, carried), (original_units, hold_carried)):
                for ticker in owned:
                    if not np.isfinite(quotes[ticker]):
                        counts[ticker] = counts.get(ticker, 0) + 1

        deposits = withdrawals = fees = 0.0
        try:
            events_today = book.execute_day(day, cash, units, values, quotes)
            for trade, before, cash, event in events_today:
                before *= fx
                if shares:
                    unit_price = before / shares
                flow, fee = event["External flow"] * fx, event["Fee"] * fx
                if flow:
                    # Recomputing the remaining shares also handles a full withdrawal exactly.
                    after_flow = before + flow
                    shares = max(0.0, after_flow) / unit_price
                    if not cash and not values:
                        shares = 0.0
                if fee:
                    attribution["Fees"] -= fee / shares
                deposits += max(flow, 0.0)
                withdrawals += max(-flow, 0.0)
                fees += fee
                net_invested += flow
                if trade is trades[0]:
                    if shares:
                        original_units = {t: q / shares for t, q in units.items()}
                        hold_cash = cash / shares
                    else:
                        original_units, hold_cash = {}, unit_price / fx
                for field in ("Sold", "Bought", "Cash after"):
                    event["Account " + field.lower()] = event[field] * fx
                    event[field] = event[field] * fx / shares if shares else 0.0
                event["External flow"], event["Fee"] = flow, fee
                snapshots.append(event)
        except MissingQuotes as exc:
            if pd.isna(exchange_rates.iloc[i]) and any(currencies.quote(t) != currencies.account for t in exc.tickers):
                raise MissingQuotes((FX_TICKER,), f"No USD/TRY rate on or before {day.date()}.") from exc
            raise
        prior_values = {t: v * fx for t, v in values.items()}
        prior_cash = cash * fx
        balance = prior_cash + fsum(prior_values.values())
        if shares:
            unit_price = balance / shares
        if day >= start:
            hold_index = (hold_cash + fsum(q * marks_today[t] for t, q in original_units.items())) * fx
            daily.append((day, unit_price, prior_cash / shares if shares else 0.0, hold_index))
            positions.append({t: v / shares for t, v in prior_values.items()} if shares else {})
            accounts.append((balance, prior_cash, deposits, withdrawals, fees, net_invested, balance - net_invested))
            contributions.append(attribution)

    ledger = pd.DataFrame(daily, columns=["Date", "Portfolio", "Cash", BUY_AND_HOLD]).set_index("Date")
    includes_funding = ledger.index[0] == calendar[0]
    scale = 1.0 if includes_funding else 100 / ledger["Portfolio"].iloc[0]
    ledger[["Portfolio", "Cash"]] *= scale
    if not includes_funding:
        ledger[BUY_AND_HOLD] *= 100 / ledger[BUY_AND_HOLD].iloc[0]
    benchmark_window = benchmark_prices.reindex(ledger.index)
    ledger["Benchmark"] = 100 * benchmark_window / benchmark_window.iloc[0]
    assets = pd.DataFrame(positions, index=ledger.index, columns=tickers, dtype=float).fillna(0) * scale
    account = pd.DataFrame(accounts, index=ledger.index, columns=[
        "Balance", "Cash", "Deposits", "Withdrawals", "Fees", "Net contributed", "Investment gain",
    ])
    attribution = pd.DataFrame(contributions, index=ledger.index, columns=[*tickers, "Cash", "Fees"]).fillna(0) * scale
    if not includes_funding:
        attribution.iloc[0] = 0.0
    history = pd.DataFrame(snapshots, columns=[
        "Log row", "Requested", "Executed", "Action", "Sell order", "Buy tickers", "Sold", "Bought",
        "Cash after", "Cash %", "Holdings after", "Note", "Leg", "Close (UTC)", "Account sold", "Account bought",
        "Account cash after", "External flow", "Fee",
    ])
    no_orders = history.loc[history["Note"].ne(""),
                            ["Log row", "Requested", "Executed", "Action", "Sell order", "Buy tickers", "Note"]].copy()
    events = history.loc[history["Executed"] >= ledger.index[0]].copy() if not history.empty else history
    if not events.empty:
        events[["Sold", "Bought", "Cash after"]] *= scale
    if len(trades) > 1 and len(sessions.closes) > 1:
        # The opening order can span several sessions while other instructions are already trading.
        passive = backtest(prices, trades[:1], bench, start, end, cash_rate, initial_cash=initial_cash,
                           currencies=currencies, funding_date=first_order, valuation_symbols=shared)
        ledger[BUY_AND_HOLD] = passive.ledger["Portfolio"]
        hold_carried = passive.carried
    return Backtest(ledger[["Portfolio", "Benchmark", "Cash", BUY_AND_HOLD]], assets, events, tuple(sorted(book.pending)),
                    carried, hold_carried, account, attribution, currencies.report, no_orders,
                    ledger.index.intersection(benchmark_history.index))
