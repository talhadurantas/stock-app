"""Order execution and sale-funded purchases across exchange closes."""

from collections import defaultdict
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import fsum

import numpy as np

from sessions import venue
from transactions import CASH_ACTIONS, MissingQuotes, PortfolioError, Trade


def execute(trade: Trade, day, cash, units, values, quotes, *, reserved=0.0, buy_budget=None):
    """Execute at today's close. Values and units describe the same positions."""
    flow = 0.0
    if trade.action == "Deposit":
        flow = trade.amount
    elif trade.action == "Withdrawal":
        if trade.amount > cash - reserved and not np.isclose(trade.amount, cash - reserved, rtol=1e-12, atol=0):
            raise PortfolioError(f"Row {trade.row}: withdrawal exceeds available cash ({cash - reserved:,.2f}).")
        flow = -min(trade.amount, cash - reserved)
    fee = trade.amount if trade.action == "Fee" else trade.fee
    cash += flow
    total = cash + fsum(values.values())
    if trade.action == "Rebalance":
        requested_sales = dict.fromkeys(units, 1.0)
    else:
        fractions = trade.sell_fractions or (1.0,) * len(trade.sells)
        requested_sales = dict(zip(trade.sells, fractions))
    sells = {t: f for t, f in requested_sales.items() if f > 0}
    missing = [t for t in sells if t not in units]
    if missing:
        raise PortfolioError(f"Row {trade.row}: cannot sell an unheld position ({', '.join(missing)}).")
    sold = fsum(values[t] * fraction for t, fraction in sells.items())
    if fee and (fee > cash - reserved + sold or fee >= total):
        raise PortfolioError(f"Row {trade.row}: the fee needs cash or sale proceeds and must leave positive equity.")
    if buy_budget is not None:
        budget = buy_budget
    elif trade.action == "Buy":
        budget = max(0.0, cash - reserved - fee)
    elif trade.action == "Sell & Buy":
        budget = max(0.0, sold - fee)
    else:
        budget = max(0.0, total - reserved - fee)
    orders = {t: budget * w for t, w in zip(trade.buys, trade.fractions) if budget * w > 0}
    needs_quote = set(sells).union(orders)
    unavailable = sorted(t for t in needs_quote if not np.isfinite(quotes.get(t, np.nan)))
    if unavailable:
        raise MissingQuotes(unavailable, f"Row {trade.row}: no fresh adjusted close for {', '.join(unavailable)} on {day.date()}. Check the symbol, listing date or trading calendar.")
    for ticker, fraction in sells.items():
        if fraction == 1:
            del units[ticker]
            del values[ticker]
        else:
            units[ticker] *= 1 - fraction
            values[ticker] *= 1 - fraction
    for ticker, amount in orders.items():
        units[ticker] = units.get(ticker, 0.0) + amount / quotes[ticker]
        values[ticker] = values.get(ticker, 0.0) + amount
    spent = fsum(orders.values())
    cash = fsum((cash, sold, -spent, -fee))
    if abs(cash) <= total * 1e-12:
        cash = 0.0
    note = ""
    if sold == 0 and spent == 0 and trade.action not in CASH_ACTIONS:
        if trade.action == "Buy":
            note = "No cash available" if budget == 0 else "All buy weights are zero"
        elif trade.buys and trade.action in ("Initial", "Rebalance"):
            note = "All buy weights are zero"
        elif trade.action == "Rebalance":
            note = "Portfolio already in cash"
        elif trade.sells:
            note = "All sell percentages are zero"
    snapshot = {
        "Log row": trade.row, "Requested": trade.date, "Executed": day, "Action": trade.action,
        "Sell order": ", ".join(f"{t} {f*100:g}%" for t, f in requested_sales.items()),
        "Buy tickers": ", ".join(trade.buys),
        "Sold": sold, "Bought": spent, "Cash after": cash,
        "External flow": flow, "Fee": fee,
        "Cash %": 100 * cash / (total - fee) if total > fee else 0.0,
        "Holdings after": ", ".join(f"{t} {100*v/(total-fee):.2f}%" for t, v in sorted(values.items())) or "All cash",
        "Note": note,
    }
    return cash, snapshot


@dataclass
class Order:
    trade: Trade
    sequence: int
    reserved: float = 0.0
    buy_budget: float | None = None


class OrderBook:
    """Each row owns its unspent basket budget until its remaining legs fill."""

    def __init__(self, trades, sessions):
        self.sessions = sessions
        self.queue = []
        self.serial = count()
        self.pending = set()
        self.reserves = {}
        self.completion = {}
        possible_holdings = set()
        for i, trade in enumerate(trades):
            if trade.date > sessions.days[-1]:
                self.pending.add(trade.row)
                continue
            tickers = set(trade.buys) | set(trade.sells)
            if trade.action == "Rebalance":
                tickers |= possible_holdings
            markets = {venue(t) for t in tickers} or {sessions.cash_market}
            times = [sessions.next_close(m, trade.date) for m in markets]
            available = [t for t in times if t is not None]
            order = Order(trade, i)
            ready = min(available) if available else None
            self.completion[i] = ready
            self.push(ready, order, "start")
            possible_holdings.update(trade.buys)

    def push(self, when, order, phase, leg=None):
        if when is None:
            self.pending.add(order.trade.row)
        else:
            heappush(self.queue, (when, order.sequence, next(self.serial), order, phase, leg))

    def groups(self, tickers, weights, earliest):
        groups = defaultdict(list)
        for ticker, weight in zip(tickers, weights):
            if weight:
                when = self.sessions.next_close(venue(ticker), earliest)
                groups[when].append((ticker, weight))
        return groups

    def plan(self, when, order, cash, units):
        trade = order.trade
        if trade.action == "Rebalance":
            # A whole-account instruction needs the positions and cash from earlier open orders.
            earlier = [done for i, done in self.completion.items()
                       if i < order.sequence and (done is None or done > when)]
            if earlier:
                ready = None if None in earlier else max(earlier)
                self.completion[order.sequence] = ready
                self.push(ready, order, "start")
                return
        sells = tuple(units) if trade.action == "Rebalance" else trade.sells
        sale_groups = self.groups(sells, trade.sell_fractions or (1.0,) * len(sells), when)
        buys_after = max((stamp for stamp in sale_groups if stamp is not None), default=when)
        if None in sale_groups:
            buy_groups = {None: list(zip(trade.buys, trade.fractions))} if trade.buys else {}
        else:
            buy_groups = self.groups(trade.buys, trade.fractions, buys_after)
        stamps = sale_groups.keys() | buy_groups.keys()
        self.completion[order.sequence] = None if None in stamps else max(stamps, default=when)
        if len(stamps) <= 1:
            self.push(next(iter(stamps), when), order, "whole", trade)
            return
        if trade.action in ("Initial", "Buy", "Rebalance"):
            self.reserves[order.sequence] = order.reserved = max(0.0, cash - fsum(self.reserves.values()))
        for stamp, parts in sale_groups.items():
            tickers, weights = zip(*parts)
            last_sale = stamp == buys_after and None not in sale_groups
            fee = trade.fee if last_sale else 0.0
            leg = replace(trade, action="Sell", sells=tickers, sell_fractions=weights, buys=(), fractions=(), fee=fee)
            self.push(stamp, order, "last sale" if last_sale else "sale", leg)
        first_buy = min((stamp for stamp in buy_groups if stamp is not None), default=None)
        for stamp, parts in buy_groups.items():
            tickers, weights = zip(*parts)
            fee = trade.fee if not sale_groups and stamp == first_buy else 0.0
            leg = replace(trade, action="Buy", buys=tickers, fractions=weights, sells=(), sell_fractions=(), fee=fee)
            self.push(stamp, order, "purchase", leg)

    def execute_day(self, day, cash, units, values, quotes):
        while self.queue and self.queue[0][0].normalize() == day:
            when, _, _, order, phase, leg = heappop(self.queue)
            if phase == "start":
                self.plan(when, order, cash, units)
                continue
            before = cash + fsum(values.values())
            locked = max(0.0, fsum(self.reserves.values()) - order.reserved)
            if phase == "purchase" and order.buy_budget is None:
                order.buy_budget = max(0.0, order.reserved - leg.fee)
                order.reserved = order.buy_budget * fsum(order.trade.fractions)
            if phase in ("sale", "last sale") and order.trade.action == "Rebalance":
                held = tuple(t for t in leg.sells if t in units)
                leg = replace(leg, sells=held, sell_fractions=(1.0,) * len(held))
            cash, event = execute(leg, day, cash, units, values, quotes, reserved=locked,
                                  buy_budget=order.buy_budget if phase == "purchase" else None)
            if phase in ("sale", "last sale"):
                order.reserved += event["Sold"]
                if phase == "last sale":
                    order.buy_budget = max(0.0, order.reserved - leg.fee)
                    order.reserved = order.buy_budget * fsum(order.trade.fractions)
            elif phase == "purchase":
                order.reserved = (0.0 if when == self.completion[order.sequence]
                                  else max(0.0, order.reserved - event["Bought"]))
            if order.reserved > 1e-12:
                self.reserves[order.sequence] = order.reserved
            else:
                self.reserves.pop(order.sequence, None)
            event.update({"Action": order.trade.action, "Leg": "Order" if phase == "whole" else leg.action,
                          "Close (UTC)": when})
            yield order.trade, before, cash, event
