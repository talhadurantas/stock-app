from datetime import date
from hashlib import sha256

import pandas as pd
import streamlit as st

from results import show_results
from statement_view import open_demo, show_statement
from currencies import NATIVE, SUPPORTED, currency_settings
from market_data import analyze_scenarios, load_prices, read_price_history
from projects import SKIP, duplicate_log, read_project, save_project, skipped_rows
from portfolio import (
    ACTIONS, COLUMNS, LOG_COLUMNS, SELL_PERCENT, PortfolioError,
    export_log, import_log, parse_log, symbol,
)


st.set_page_config(page_title="Portfolio Visualizer", layout="wide")
get_prices = st.cache_data(ttl=3600, show_spinner=False)(load_prices)
EXAMPLES = {
    "Tech growth": [
        ("2013-01-01", "Initial", "", "AAPL, NVDA, MSFT", "40, 40, 20"),
        ("2017-02-05", "Sell & Buy", "MSFT", "INTC", "100"),
        ("2020-03-15", "Buy", "", "GOOGL", "50"),
        ("2022-01-10", "Rebalance", "", "AAPL, NVDA, GOOGL, TSLA", "25, 25, 25, 25"),
    ],
    "Value play": [
        ("2013-01-01", "Initial", "", "BRK.B, JPM, JNJ", "50, 25, 25"),
        ("2018-06-15", "Buy", "", "V", "100"),
        ("2021-09-20", "Sell & Buy", "BRK.B", "META", "80"),
    ],
}


def table(rows):
    frame = pd.DataFrame(rows, columns=COLUMNS, dtype="string")
    return frame.reindex(columns=LOG_COLUMNS, fill_value="")


def replace_draft(frame, message):
    st.session_state.editor_seed = frame.reset_index(drop=True)
    st.session_state.editor_version += 1
    st.session_state.editor_message = message
    st.rerun()


def edit_log(frame, key, scenario=False):
    order = ["Date", "Action", "Sell", SELL_PERCENT, "Buy", "Weights", "Amount", "Fee"]
    return st.data_editor(
        frame, key=key, num_rows="dynamic", hide_index=True, width="stretch",
        column_order=[SKIP, *order] if scenario else order,
        column_config={
            SKIP: st.column_config.CheckboxColumn("Skip", default=False),
            "Date": st.column_config.TextColumn("Date", help="YYYY-MM-DD or YYYYMMDD"),
            "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS),
            "Sell": st.column_config.TextColumn("Sell tickers"),
            SELL_PERCENT: st.column_config.TextColumn("Sell % of position", help="Blank sells the whole position."),
            "Buy": st.column_config.TextColumn("Buy tickers"),
            "Weights": st.column_config.TextColumn("Buy weights %"),
            "Amount": st.column_config.TextColumn("Amount", help="Deposit, withdrawal or standalone fee in account currency."),
            "Fee": st.column_config.TextColumn("Fee", help="Flat fee for this row, in account currency."),
        },
    )


def editor():
    state = st.session_state
    if "editor_seed" not in state:
        state.editor_seed = table(EXAMPLES["Tech growth"])
        state.editor_version = 0
    st.subheader("Transaction log")
    if "editor_message" in state:
        st.success(state.pop("editor_message"))
    with st.expander("Transaction rules"):
        st.markdown(
            "**Initial** and **Rebalance** allocate percentages of total value. "
            "Leave Buy empty to hold everything in cash. **Buy** spends percentages of available cash. "
            "**Sell** and **Sell & Buy** use the Sell % of each named position. "
            "Enter 25 to sell one quarter, or leave the field blank to sell in full. "
            "**Sell & Buy** applies the buy weights to those sale proceeds. "
            "List percentages in ticker order. Blank buy weights split the available budget equally. "
            "**Deposit**, **Withdrawal** and **Fee** take a positive Amount in account currency. "
            "An optional flat Fee on a trade is reserved before purchases. Withdrawals use available cash."
        )
    edited = edit_log(state.editor_seed.reindex(columns=LOG_COLUMNS, fill_value=""), f"log_{state.editor_version}")
    add, clear, tech, value = st.columns(4)
    if add.button("Add row", width="stretch"):
        row = table([(date.today().isoformat(), "Buy", "", "", "")])
        replace_draft(pd.concat([edited, row], ignore_index=True), "Added a row.")
    if clear.button("Clear log", width="stretch"):
        replace_draft(table([]), "Cleared the transaction log.")
    for control, name in ((tech, "Tech growth"), (value, "Value play")):
        if control.button(name + " example", width="stretch"):
            replace_draft(table(EXAMPLES[name]), f"Loaded the {name.lower()} example.")
    with st.expander("Import or export transaction CSV"):
        uploaded = st.file_uploader("Transaction CSV", type=["csv"], key="csv_file",
                                    help="Use a transaction CSV exported by this or the previous app.")
        use_file, download = st.columns(2)
        if use_file.button("Replace log with CSV", disabled=uploaded is None, key="import_log"):
            try:
                imported = import_log(uploaded.getvalue())
            except PortfolioError as exc:
                st.error(str(exc))
            else:
                replace_draft(imported, f"Imported {len(imported)} transaction rows.")
        download.download_button("Download transaction CSV", export_log(edited),
                                 "portfolio_transactions.csv", "text/csv", on_click="ignore", key="log_csv")
    return edited


def scenario_editor(original):
    state = st.session_state
    enabled = st.toggle("Compare a what-if scenario", key="compare_scenario")
    if not enabled:
        if state.pop("scenario_was_open", False) and "scenario_draft" in state:
            state.scenario_seed = state.scenario_draft.copy()
            state.scenario_version += 1
        return None
    if "scenario_seed" not in state:
        state.scenario_seed = duplicate_log(original)
        state.scenario_version = 0
    state.scenario_was_open = True
    st.subheader("What-if scenario")
    st.caption("Skip a row or change its details. Both logs use the original funding date, prices and settings.")
    if st.button("Copy current original into scenario", key="duplicate"):
        state.scenario_seed = duplicate_log(original)
        state.scenario_version += 1
    draft = edit_log(state.scenario_seed, f"variant_{state.scenario_version}", scenario=True)
    state.scenario_draft = draft.copy()
    return draft


def restore_project(filename, content):
    state = st.session_state
    try:
        original, alternate, settings = read_project(content)
    except PortfolioError as exc:
        state.project_error = str(exc)
        return
    if settings["report_currency"] == NATIVE:
        settings["report_currency"] = "USD"
    hurdle = settings.pop("risk_free_rate", 2.0)
    report = NATIVE if settings["account_currency"] == NATIVE else settings["report_currency"]
    state[f"hurdle_{report}"] = hurdle
    settings.setdefault("price_source", "Yahoo Finance")
    state.price_upload_version = state.get("price_upload_version", 0) + 1
    state.update(settings)
    state.editor_seed = original
    state.editor_version = state.get("editor_version", 0) + 1
    state.compare_scenario = alternate is not None
    state.scenario_seed = alternate if alternate is not None else duplicate_log(original)
    state.scenario_draft = state.scenario_seed.copy()
    state.scenario_version = state.get("scenario_version", 0) + 1
    state.scenario_was_open = alternate is not None
    state.pop("analysis", None)
    state.pop("project_error", None)
    state.last_opened_project = filename


def project_controls(original, alternate, settings):
    with st.expander("Save and reopen comparisons"):
        uploaded = st.file_uploader("Saved comparison JSON", type=["json"], key="project_file")
        filename, content = (uploaded.name, uploaded.getvalue()) if uploaded is not None else ("", b"")
        st.button(f"Open {filename}" if filename else "Open saved comparison", disabled=not filename,
                  on_click=restore_project, args=(filename, content), key="open_project")
        if "last_opened_project" in st.session_state:
            st.caption(f"Last opened {st.session_state.last_opened_project}")
        if "project_error" in st.session_state:
            st.error(st.session_state.project_error)
        if alternate is not None:
            st.download_button("Save both logs and settings", save_project(original, alternate, settings),
                               "portfolio_comparison.json", "application/json", key="save_project", on_click="ignore")


def main():
    st.title("Portfolio Visualizer")
    state = st.session_state
    state.setdefault("workspace", "Brokerage statement" if st.query_params.get("demo") == "statement" else "Transaction log")
    workspace = st.sidebar.radio("Workspace", ["Transaction log", "Brokerage statement"], key="workspace")
    if workspace == "Brokerage statement":
        show_statement()
        return
    st.button("Try the offline statement demo", on_click=open_demo, key="offline_demo")
    st.caption("Compare your transactions with holding your original investments and with a market benchmark.")
    for key, default in dict(start_date=date(2013, 1, 1), end_date=date.today(), benchmark="SPY",
                             cash_rate=4.5, starting_cash=100.0).items():
        state.setdefault(key, default)
    with st.sidebar:
        st.header("Analysis settings")
        start = st.date_input("Start date", min_value=date(1900, 1, 1),
                              max_value=date.today(), key="start_date")
        end = st.date_input("End date", min_value=date(1900, 1, 1),
                            max_value=date.today(), key="end_date")
        source = st.selectbox("Price history", ["Yahoo Finance", "Adjusted-close CSV"], key="price_source")
        uploaded_prices = None
        if source == "Adjusted-close CSV":
            uploaded_prices = st.file_uploader("Price history CSV", type="csv",
                                               key=f"prices_{state.get('price_upload_version', 0)}")
            st.caption("Use Date, Ticker, Adjusted close. Include native prices for the holdings and benchmark, "
                       "plus TRY=X for USD/TRY conversions. Reattach this file when reopening a comparison.")
        bench = symbol(st.text_input("Benchmark ticker", key="benchmark"))
        rate = st.number_input("Annual cash interest (%)", min_value=0.0, max_value=100.0,
                               step=0.25, key="cash_rate")
        initial_cash = st.number_input("Starting cash", min_value=0.0, step=100.0, key="starting_cash")
        account_ccy = st.selectbox("Account currency", [*SUPPORTED, NATIVE], key="account_currency",
                                   help="Currency of cash, deposits, withdrawals and fees. As quoted replays a single-currency log.")
        report_ccy = st.selectbox("Report in", SUPPORTED, disabled=account_ccy == NATIVE, key="report_currency")
        report = NATIVE if account_ccy == NATIVE else report_ccy
        key = f"hurdle_{report}"
        state.setdefault(key, state.get("hurdles", {}).get(report, rate if account_ccy == report else 0.0))
        hurdle = st.number_input(f"Annual risk-free rate (%) · {report}", min_value=-99.99, step=0.25, key=key,
                                 help="Annual assumption for Sharpe and Sortino in the reporting currency.")
        state.setdefault("hurdles", {})[report] = hurdle
        with st.expander("Ticker currencies"):
            st.caption("Istanbul (.IS) listings use TRY. US symbols use USD. Declare other USD or TRY listings below.")
            overrides = st.text_input("Quote currency overrides", key="currency_overrides", placeholder="XYZ=TRY, ABC=USD")
        st.subheader("Chart settings")
        markers = st.checkbox("Mark trades", True)
        shade = st.checkbox("Highlight portfolio max drawdown", False)
        compare = st.checkbox("Show benchmark on rolling charts", True)
        logarithmic = st.checkbox("Logarithmic growth chart", False, key="log_chart")
        with st.expander("Valuation assumptions"):
            st.write("Each trade executes at the next close of its own exchange. "
                     "Sales fund purchases at the same or a later exchange close. Ties follow log order. "
                     "Prices include split and dividend adjustments. Cash compounds once per account-market session, "
                     "using 252 sessions per year. Prices and the benchmark use the last available USD/TRY close "
                     "on or before each valuation date when conversion is needed. Trades use fractional positions. "
                     "Fees are charged as entered. Deposits and withdrawals change the account balance at the execution close.")
    edited = editor()
    alternate = scenario_editor(edited)
    settings = dict(start_date=start, end_date=end, benchmark=bench, cash_rate=rate, starting_cash=initial_cash,
                    account_currency=account_ccy, report_currency=report_ccy, currency_overrides=overrides, risk_free_rate=hurdle,
                    price_source=source)
    if start is not None and end is not None:
        project_controls(edited, alternate, settings)
    scenario_trades = None
    scenario_error = None
    if alternate is not None:
        try:
            scenario_trades = parse_log(alternate, skip_rows=skipped_rows(alternate), allow_empty=True)
        except PortfolioError as exc:
            scenario_error = str(exc)
            st.warning("What-if scenario: " + scenario_error)
    error = None
    try:
        trades = parse_log(edited)
        if source == "Adjusted-close CSV" and uploaded_prices is None:
            raise PortfolioError("Attach a price history CSV for this analysis.")
        currencies = currency_settings(account_ccy, report_ccy, overrides)
        if not bench or "," in bench or any(c.isspace() for c in bench):
            raise PortfolioError("Enter one benchmark ticker.")
        if start is None or end is None or start > end:
            raise PortfolioError("Choose an end date on or after the start date.")
        if trades[0].date > pd.Timestamp(end):
            raise PortfolioError("The first transaction is after the end date.")
    except PortfolioError as exc:
        error = str(exc)
        st.warning(error)
    variant_signature = None if alternate is None else (export_log(alternate), tuple(sorted(skipped_rows(alternate))))
    price_bytes = uploaded_prices.getvalue() if uploaded_prices is not None else None
    price_digest = sha256(price_bytes).digest() if price_bytes is not None else None
    signature = (4, export_log(edited), variant_signature, price_digest, *settings.values())
    if st.button("Run analysis", key="run", type="primary", disabled=error is not None):
        st.session_state.pop("analysis", None)
        try:
            with st.spinner("Loading adjusted prices and replaying transactions…"):
                snapshot = read_price_history(price_bytes) if price_bytes is not None else None
                results = analyze_scenarios(trades, scenario_trades, bench, start, end, rate, initial_cash, currencies,
                                            cached_download=get_prices, clear_cache=get_prices.clear, price_history=snapshot)
            original, variant, replay_error = results
            st.session_state.analysis = (signature, (original, variant, scenario_error or replay_error))
        except PortfolioError as exc:
            st.error(str(exc))
    previous = st.session_state.get("analysis")
    if previous is not None:
        previous_signature, results = previous
        if previous_signature == signature:
            show_results(*results, bench, rate, hurdle / 100, markers, shade, compare, logarithmic, edited, source)
        else:
            st.info("Inputs changed. Run analysis to update your results.")


if __name__ == "__main__":
    main()
