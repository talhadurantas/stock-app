"""The offline example and the user-supplied brokerage statement workflow."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics import drawdown
from currencies import SUPPORTED
from statements import read_balances, read_closes, read_fills, reconcile, replay_statement
from transactions import PortfolioError


EXAMPLE = Path(__file__).parent / "examples" / "statement"
FILES = {"fills.csv": "Activity CSV", "closes.csv": "Unadjusted closes CSV", "balances.csv": "Closing balances CSV"}


def open_demo():
    st.session_state.workspace = "Brokerage statement"
    st.session_state.statement_source = "Synthetic example"


def show_statement():
    st.subheader("Brokerage statement replay")
    source = st.radio("Statement source", ["Synthetic example", "Upload my CSVs"], horizontal=True, key="statement_source")
    sample = source == "Synthetic example"
    with st.expander("CSV templates and conventions"):
        st.write("Activity timestamps include a UTC offset. Enter each fill separately. Use the booked price and actual "
                 "share count, and enter fees in account currency. FX is account currency per unit of quote currency. "
                 "Deposit, Withdrawal, Dividend and Fee use Amount in account currency. Split uses Quantity as new shares "
                 "per old share. Enter booked interest as Dividend (cash income).")
        st.write("Daily marks use unadjusted closes and UTC dates. Supply an unadjusted close on each day a fill or "
                 "split leaves a position open. Other dates carry the latest supplied mark. For positions carried "
                 "across an FX move, supply the carried native close and that day's FX. The closing balances file "
                 "lists all shares and cash on one date. Both the original and the scenario stop on that date.")
        for column, (name, label) in zip(st.columns(3), FILES.items()):
            column.download_button(label + " template", (EXAMPLE / name).read_bytes(), name,
                                   "text/csv", key="template_" + name, on_click="ignore")
    if sample:
        st.info("Synthetic US and Istanbul example. The prices, activity and closing balances are fictional. "
                "This demo runs entirely from the included CSVs.")
        account, opening = "USD", 0.0
        activity, prices, balances = [(EXAMPLE / name).read_bytes() for name in FILES]
    else:
        account = st.selectbox("Statement account currency", SUPPORTED, key="statement_currency")
        opening = st.number_input("Cash before the first activity", min_value=0.0, key="statement_opening")
        st.caption("Start with an all-cash account or include activity from inception. Supply a single cash "
                   "balance and the FX actually booked for foreign trades.")
        uploads = [st.file_uploader(label, type="csv", key="statement_" + name) for name, label in FILES.items()]
        if any(file is None for file in uploads):
            st.info("Upload activity, daily closes and closing balances to check the original account.")
            return
        activity, prices, balances = [file.getvalue() for file in uploads]
    try:
        fills = read_fills(activity, account)
        closes = read_closes(prices, fills, account)
        end, expected = read_balances(balances)
        original = replay_statement(fills, closes, opening, end)
    except PortfolioError as exc:
        st.error(str(exc))
        return
    matched = reconcile(original, expected)
    st.subheader("Statement reconciliation")
    st.caption(f"Closing cash in {account} and shares as of {end.date()}.")
    if matched["Match"].all():
        st.success("Original cash and all share quantities match the closing statement.")
    else:
        st.warning("The original replay differs from the closing statement. Review the differences below.")
    st.dataframe(matched, hide_index=True, width="stretch")
    with st.expander("Recorded activity"):
        st.dataframe(pd.DataFrame([asdict(f) for f in fills if f.day <= end]), hide_index=True)
    choices = {f.line: f for f in fills if f.day <= end and f.action in ("Buy", "Sell")}
    def label(line):
        fill = choices[line]
        return f"Line {line} · {fill.stamp:%Y-%m-%d %H:%M} UTC · {fill.action} {fill.quantity:g} {fill.ticker} at {fill.price:g} {fill.currency}"
    st.subheader("What if I skipped a trade?")
    # A new upload gets independent skip selections, even if both files contain the same CSV line numbers.
    skipped = st.multiselect("Skip filled trades", list(choices), default=[5] if sample else [], format_func=label,
                             key="skip_fills_" + sha256(activity).hexdigest()[:20])
    st.caption("Skipping a fill also skips its fee. Other trades keep their recorded quantities, prices and FX. "
               "Cash income stays as booked. Daily performance adjusts for deposits and withdrawals.")
    views = {"Original": original}
    try:
        if skipped:
            views["What-if"] = replay_statement(fills, closes, opening, end, skipped=skipped)
    except PortfolioError as exc:
        st.warning(f"What-if scenario: {exc}")
    summary = pd.DataFrame([{"History": name, f"Ending balance ({account})": view.daily.Balance.iloc[-1],
                             "Ending value per 100": view.daily["Performance per 100"].iloc[-1],
                             "Worst drawdown (%)": 100 * drawdown(view.daily["Performance per 100"], 100)[0].min()}
                            for name, view in views.items()]).set_index("History")
    if "What-if" in views:
        diff = summary.loc["What-if"] - summary.loc["Original"]
        left, right = st.columns(2)
        left.metric("Change in ending balance", f"{diff.iloc[0]:+,.2f} {account}")
        right.metric("Change in worst drawdown", f"{diff.iloc[2]:+.2f} pp")
    st.dataframe(summary.round(3), width="stretch")
    curves = pd.DataFrame({k: v.daily["Performance per 100"] for k, v in views.items()})
    st.line_chart(curves, y_label="Performance per 100")
    ledger = pd.concat({k: v.daily for k, v in views.items()}, axis=1)
    ledger.columns = [" ".join(column) for column in ledger.columns]
    st.download_button("Download statement replay", ledger.to_csv(index_label="Date"), "statement_replay.csv",
                       "text/csv", key="statement_ledger", on_click="ignore")
