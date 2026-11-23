from calendar import month_abbr
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analytics import PERCENT_METRICS, RISK_FREE_RATE, daily_returns, drawdown, holding_breakdown, monthly_returns, rolling_metrics, stats
from charts import growth_chart
from currencies import NATIVE, SUPPORTED, currency_settings
from market_data import analyze_scenarios, load_prices
from projects import SKIP, duplicate_log, read_project, save_project, skipped_rows
from portfolio import (
    ACTIONS, BUY_AND_HOLD, COLUMNS, LOG_COLUMNS, SELL_PERCENT, PortfolioError,
    export_log, import_log, parse_log, symbol,
)


st.set_page_config(page_title="Portfolio Visualizer", layout="wide")
get_prices = st.cache_data(ttl=3600, show_spinner=False)(load_prices)
BLUE, ORANGE = "#176a78", "#a75020"
PERFORMANCE_NAMES = {"Portfolio": "Portfolio Value", "Benchmark": "Benchmark Value", "Cash": "Cash Value"}
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


def restore_project():
    state = st.session_state
    try:
        original, alternate, settings = read_project(state.project_file.getvalue())
    except PortfolioError as exc:
        state.project_error = str(exc)
        return
    if settings["report_currency"] == NATIVE:
        settings["report_currency"] = "USD"
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
    state.editor_message = "Opened the saved comparison. Run analysis to replay it."


def project_controls(original, alternate, settings):
    with st.expander("Save or open a comparison"):
        uploaded = st.file_uploader("Saved comparison JSON", type=["json"], key="project_file")
        st.button("Open saved comparison", disabled=uploaded is None, on_click=restore_project, key="open_project")
        if "project_error" in st.session_state:
            st.error(st.session_state.project_error)
        if alternate is not None:
            st.download_button("Save both logs and settings", save_project(original, alternate, settings),
                               "portfolio_comparison.json", "application/json", key="save_project", on_click="ignore")


def formatted(name, value):
    if not np.isfinite(value):
        return "—"
    return f"{value:.2%}" if name in PERCENT_METRICS else f"{value:,.2f}"


def show_no_orders(rows):
    if rows.empty:
        return
    if len(rows) == 1:
        row = rows.iloc[0]
        tickers = " / ".join(filter(None, (row["Sell order"], row["Buy tickers"])))
        detail = f"{row['Action']} {tickers}".strip()
        st.warning(f"Row {row['Log row']} · {row['Executed']:%d %b %Y} · {detail} placed no orders. {row['Note']}.")
    else:
        st.warning(f"{len(rows)} transactions placed no orders.")
        detail = rows.assign(Requested=rows["Requested"].dt.date, Executed=rows["Executed"].dt.date)
        st.dataframe(detail.rename(columns={"Note": "Reason"}), hide_index=True)


def risk_charts(result, bench, compare_rolling):
    portfolio = result.ledger["Portfolio"]
    returns = daily_returns(portfolio, 100).dropna()
    if returns.empty:
        st.info("A second closing valuation is needed for return distributions.")
        return
    with st.expander("Monthly returns"):
        months = monthly_returns(portfolio, 100)
        grid = months.to_frame("Return").assign(Year=months.index.year, Month=months.index.month)
        pivot = grid.pivot(index="Year", columns="Month", values="Return").reindex(columns=range(1, 13))
        fig = Figure(figsize=(10, max(2, 0.40 * len(pivot) + 1.2)), layout="constrained")
        ax = fig.subplots()
        color_scale = max(0.10, np.nanpercentile(np.abs(pivot.values), 90))
        ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu", vmin=-color_scale, vmax=color_scale)
        ax.set_xticks(range(12), list(month_abbr)[1:])
        ax.set_yticks(range(len(pivot)), [str(y) for y in pivot.index])
        for i, row in enumerate(pivot.to_numpy()):
            for j, number in enumerate(row):
                if np.isfinite(number):
                    ax.text(j, i, f"{number:.1%}", ha="center", va="center", fontsize=8,
                            color="white" if abs(number) > 0.62 * color_scale else "#17212b")
        st.pyplot(fig)
        monthly_stats = st.columns(4)
        for column, name, n in zip(monthly_stats, ["Average month", "Median month", "Best month", "Worst month"],
                                   [months.mean(), months.median(), months.max(), months.min()]):
            column.metric(name, f"{n:.2%}")
        st.caption(f"Positive months: {months.gt(0).sum()} of {len(months)}. Edge months include the observed portion only.")
    with st.expander("Rolling one-year metrics"):
        if len(returns) < 252:
            st.info("This chart needs 252 close-to-close returns.")
        else:
            fig = Figure(figsize=(10, 7), layout="constrained")
            axes = fig.subplots(3, 1, sharex=True)
            columns = ["Portfolio", "Benchmark"] if compare_rolling else ["Portfolio"]
            for name, color in zip(columns, [BLUE, ORANGE]):
                series = rolling_metrics(result.ledger[name]).iloc[252:]
                for ax, kind in zip(axes, series):
                    ax.plot(series.index, series[kind], color=color, linewidth=1.2,
                            label=bench if name == "Benchmark" else "Portfolio")
                    ax.set_ylabel(kind)
                    if kind != "Sharpe":
                        ax.yaxis.set_major_formatter(PercentFormatter(1))
            for ax in axes:
                ax.grid(axis="y", alpha=0.2)
            axes[0].legend(frameon=False)
            st.pyplot(fig)
    with st.expander("Daily return distribution and tail risk"):
        fig = Figure(figsize=(10, 3.1), layout="constrained")
        hist, cdf = fig.subplots(1, 2)
        hist.hist(returns, bins=min(50, max(1, int(np.sqrt(len(returns))))), color=BLUE, alpha=0.85)
        hist.axvline(returns.quantile(0.05), color=ORANGE, linestyle="--", label="5th percentile")
        hist.legend(frameon=False)
        hist.set_ylabel("Sessions")
        cdf.step(np.sort(returns), np.arange(1, len(returns)+1)/len(returns), where="post", color=BLUE)
        cdf.set_ylabel("Cumulative share")
        cdf.yaxis.set_major_formatter(PercentFormatter(1))
        for ax in [hist, cdf]:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("Daily return")
        st.pyplot(fig)
        st.caption("The 5% and 1% return quantiles and the corresponding average tail returns appear in Detailed statistics.")


def show_results(result, alternative, scenario_error, bench, cash_rate, markers, shade, compare_rolling, logarithmic, draft):
    ledger = result.ledger
    first, last = ledger.index[0].date(), ledger.index[-1].date()
    if alternative is not None:
        ledger = ledger.assign(Scenario=alternative.ledger["Portfolio"])
    strategies = ["Portfolio", BUY_AND_HOLD, "Benchmark"]
    if alternative is not None:
        strategies.insert(1, "Scenario")
    summary = pd.DataFrame({name: stats(ledger[name], opening=100) for name in strategies})
    summary.loc["Ending value per 100"] = ledger.iloc[-1].reindex(strategies)
    pretty = pd.DataFrame({name: [formatted(metric, v) for metric, v in col.items()]
                           for name, col in summary.items()}, index=summary.index)
    pretty = pretty.rename(columns={"Benchmark": bench + " (benchmark)"})
    port, reference = summary["Portfolio"], summary["Benchmark"]
    spread = port["Total return"] - reference["Total return"]
    st.divider()
    st.header("Performance")
    st.caption(f"{first:%d %b %Y} – {last:%d %b %Y} · {len(ledger):,} benchmark closes")
    top = st.columns(4)
    top[0].metric("Portfolio return", formatted("Total return", port["Total return"]))
    top[1].metric(f"{bench} return", formatted("Total return", reference["Total return"]))
    top[2].metric("Return difference", f"{spread*100:+.2f} pp")
    top[3].metric("Portfolio max drawdown", formatted("Max drawdown", port["Max drawdown"]))
    show_no_orders(result.no_orders)
    if scenario_error:
        st.error("What-if scenario: " + scenario_error)
    st.subheader("Strategy comparison")
    st.caption("Buy and hold keeps the positions from your first transaction. Any cash left earns the same "
               "interest rate. Performance tracks an opening investment of 100 and adjusts for deposits and withdrawals.")
    comparison = pretty.loc[["Ending value per 100", "Max drawdown"]].T.rename_axis("Strategy")
    st.dataframe(comparison.reset_index(), hide_index=True)
    if alternative is not None:
        gap = summary.at["Ending value per 100", "Scenario"] - summary.at["Ending value per 100", "Portfolio"]
        dd = 100 * (summary.at["Max drawdown", "Scenario"] - summary.at["Max drawdown", "Portfolio"])
        st.write(f"Scenario minus original: **{gap:+,.2f}** per 100 in ending value and **{dd:+.2f} pp** in maximum drawdown.")
    st.altair_chart(growth_chart(result, alternative, markers=markers, shade=shade, logarithmic=logarithmic),
                    width="stretch", theme=None)
    st.caption(f"Drag or scroll to zoom the time axis. Hover over a trade marker for orders and cash in {result.currency}.")
    run_views = {"Portfolio": result}
    if alternative is not None:
        run_views["Scenario"] = alternative
    st.subheader("Account value and investment gain")
    account_rows = []
    for label, view in run_views.items():
        last_account = view.account.iloc[-1]
        account_rows.append({"Strategy": label, "Balance": last_account["Balance"],
                             "Net contributed": last_account["Net contributed"], "Investment gain": last_account["Investment gain"],
                             "Cash": last_account["Cash"], "Fees in window": view.account["Fees"].sum()})
    st.dataframe(pd.DataFrame(account_rows).round(2), hide_index=True)
    st.caption(f"Amounts in {result.currency}. Net contributed includes starting cash and flows since funding, each valued on its own date.")
    with st.expander("Return by holding and worst drawdown", expanded=True):
        for label, view in run_views.items():
            detail = holding_breakdown(view).sort_values("Total return (pp)", ascending=False)
            detail.loc["Total"] = detail.sum()
            if alternative is not None:
                st.markdown(f"**{label}**")
            st.dataframe(detail.round(3))
        st.caption("Each column adds up to the strategy's return or maximum drawdown in percentage points. Cash includes interest and currency changes.")
    if result.pending:
        st.info("Rows outside the available closing dates: " + ", ".join(map(str, result.pending)) + ".")
    for title, carried in (("Portfolio", result.carried), (BUY_AND_HOLD, result.hold_carried)):
        if carried:
            details = ", ".join(f"{ticker} ({n})" for ticker, n in sorted(carried.items()))
            st.warning(f"{title} used a previous close to value these holdings (number of sessions): {details}.")
    with st.expander("Holdings and transaction history", expanded=True):
        allocations = result.positions.iloc[-1]
        holdings = [(ticker, value) for ticker, value in allocations.items() if value > 0]
        holdings.append(("Cash", ledger["Cash"].iloc[-1]))
        allocation = pd.DataFrame(holdings, columns=["Position", "Closing value"])
        allocation["Weight %"] = allocation["Closing value"] / ledger["Portfolio"].iloc[-1] * 100
        allocation["Closing value"] *= result.account["Balance"].iloc[-1] / ledger["Portfolio"].iloc[-1]
        st.markdown(f"**Closing positions on {last:%d %b %Y}**")
        st.dataframe(allocation.round(3), hide_index=True)
        st.markdown("**Executed transactions in the selected window**")
        if result.executions.empty:
            st.caption("Positions carried in from earlier transactions.")
        else:
            events = result.executions.copy()
            events["Requested"] = events["Requested"].dt.date
            events["Executed"] = events["Executed"].dt.date
            st.dataframe(events.round(3), hide_index=True)
        if alternative is not None:
            st.markdown("**Scenario transactions**")
            if not alternative.executions.empty:
                st.dataframe(alternative.executions.round(3), hide_index=True)
            if not alternative.no_orders.empty:
                show_no_orders(alternative.no_orders)
            if alternative.pending:
                st.info("Pending scenario rows: " + ", ".join(map(str, alternative.pending)))
            if alternative.carried:
                st.caption(f"Scenario positions using previous closes (sessions): {alternative.carried}")
        cash_weight = result.account["Cash"].div(result.account["Balance"].replace(0, np.nan))
        st.caption(f"Average session-end allocation: {1-cash_weight.mean():.1%} invested and {cash_weight.mean():.1%} cash.")
    with st.expander("Detailed statistics"):
        st.dataframe(pretty)
        for name in strategies:
            title = bench if name == "Benchmark" else name
            values = ledger[name]
            declines, peak, trough = drawdown(values, opening=100)
            if declines.min() < 0:
                st.caption(f"{title} drawdown peak {peak.date()} · trough {trough.date()} · {(trough - peak).days} calendar days")
            changes = daily_returns(values, 100).dropna()
            if not changes.empty:
                st.caption(f"{title} best day {changes.idxmax().date()} · worst day {changes.idxmin().date()}")
        st.caption(f"Risk metrics use 252 sessions per year and a {RISK_FREE_RATE:.0%} annual risk-free assumption. A dash indicates an undefined metric.")
    risk_charts(result, bench, compare_rolling)
    with st.expander("Download results"):
        out = ledger[list(PERFORMANCE_NAMES)].rename(columns=PERFORMANCE_NAMES).copy()
        out["Portfolio Daily Return"] = daily_returns(ledger["Portfolio"], 100)
        out["Benchmark Daily Return"] = ledger["Benchmark"].pct_change(fill_method=None)
        out = out.join(result.positions.add_prefix("Position value: "))
        out["Buy and Hold Value"] = ledger[BUY_AND_HOLD]
        out["Buy and Hold Daily Return"] = daily_returns(ledger[BUY_AND_HOLD], 100)
        out = out.join(result.account.add_prefix("Account: ")).join(result.contributions.add_prefix("Return points: "))
        if alternative is not None:
            out["Scenario Value"] = ledger["Scenario"]
            out = out.join(alternative.account.add_prefix("Scenario account: "))
            out = out.join(alternative.contributions.add_prefix("Scenario return points: "))
        report = "\n".join([
            "PORTFOLIO VISUALIZER", f"{first} to {last}", f"Benchmark: {bench}",
            f"Cash rate: {cash_rate:.2f}% per year, compounded each benchmark session.",
            f"Account amounts in {result.currency}. Performance per 100 is adjusted for external cash flows.",
            pd.DataFrame(account_rows).to_string(index=False),
            *(label + " holding contributions (pp)\n" + holding_breakdown(view).to_string() for label, view in run_views.items()),
            *(["Scenario execution history\n" + alternative.executions.to_string(index=False)] if alternative is not None else []),
            *(["Scenario error: " + scenario_error] if scenario_error else []),
            "Buy and hold keeps the positions and cash left by the first transaction.",
            f"Portfolio minus benchmark: {spread * 100:+.2f} percentage points.",
            f"Average cash allocation: {cash_weight.mean():.2%}", "", pretty.to_string(),
            "", "Executed transactions in the reporting window",
            result.executions.to_string(index=False) if not result.executions.empty else "None in the window.",
            "", "Transactions that placed no orders (including earlier rows replayed for the opening holdings)",
            result.no_orders.to_string(index=False) if not result.no_orders.empty else "None",
            "", "Unexecuted row numbers: " + (", ".join(map(str, result.pending)) or "None"),
            "Carried-forward marks (ticker and number of sessions): " + (str(result.carried) if result.carried else "None"),
            "Buy-and-hold carried-forward marks: " + (str(result.hold_carried) if result.hold_carried else "None"),
        ])
        left, middle, right = st.columns(3)
        left.download_button("Daily ledger CSV", out.to_csv(index_label="Date"), f"portfolio_{first}_{last}.csv",
                             "text/csv", on_click="ignore", key="ledger_csv")
        middle.download_button("Transaction CSV", export_log(draft), "portfolio_transactions.csv",
                               "text/csv", on_click="ignore", key="result_log")
        right.download_button("Summary report", report, f"portfolio_report_{first}_{last}.txt",
                              "text/plain", on_click="ignore", key="report_txt")


def main():
    st.title("Portfolio Visualizer")
    st.caption("Compare your transactions with holding your original investments and with a market benchmark.")
    with st.sidebar:
        st.header("Analysis settings")
        start = st.date_input("Start date", date(2013, 1, 1), min_value=date(1900, 1, 1),
                              max_value=date.today(), key="start_date")
        end = st.date_input("End date", date.today(), min_value=date(1900, 1, 1),
                            max_value=date.today(), key="end_date")
        bench = symbol(st.text_input("Benchmark ticker", "SPY", key="benchmark"))
        rate = st.number_input("Annual cash interest (%)", min_value=0.0, max_value=100.0,
                               value=4.5, step=0.25, key="cash_rate")
        initial_cash = st.number_input("Starting cash", min_value=0.0, value=100.0, step=100.0, key="starting_cash")
        account_ccy = st.selectbox("Account currency", [*SUPPORTED, NATIVE], key="account_currency",
                                   help="Currency of cash, deposits, withdrawals and fees. As quoted replays a single-currency log.")
        report_ccy = st.selectbox("Report in", SUPPORTED, disabled=account_ccy == NATIVE, key="report_currency")
        with st.expander("Ticker currencies"):
            st.caption("Istanbul (.IS) listings use TRY. US symbols use USD. Declare other USD or TRY listings below.")
            overrides = st.text_input("Quote currency overrides", key="currency_overrides", placeholder="XYZ=TRY, ABC=USD")
        st.subheader("Chart settings")
        markers = st.checkbox("Mark trades", True)
        shade = st.checkbox("Highlight portfolio max drawdown", False)
        compare = st.checkbox("Show benchmark on rolling charts", True)
        logarithmic = st.checkbox("Logarithmic growth chart", False, key="log_chart")
        with st.expander("Valuation assumptions"):
            st.write("Trades execute at the next benchmark close on or after the date entered. "
                     "Earlier holdings earn the move into that close. Same-date trades follow log order. "
                     "Prices include split and dividend adjustments. Cash compounds once per benchmark session, "
                     "using 252 sessions per year. Prices and the benchmark use the last available USD/TRY close "
                     "on or before each valuation date when conversion is needed. Trades use fractional positions. "
                     "Fees are charged as entered. Deposits and withdrawals change the account balance at the execution close.")
    edited = editor()
    alternate = scenario_editor(edited)
    settings = dict(start_date=start, end_date=end, benchmark=bench, cash_rate=rate, starting_cash=initial_cash,
                    account_currency=account_ccy, report_currency=report_ccy, currency_overrides=overrides)
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
        currencies = currency_settings(account_ccy, report_ccy, overrides)
        if initial_cash <= 0:
            raise PortfolioError("Starting cash must be positive.")
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
    signature = (3, export_log(edited), variant_signature, *settings.values())
    if st.button("Run analysis", key="run", type="primary", disabled=error is not None):
        st.session_state.pop("analysis", None)
        try:
            with st.spinner("Loading adjusted prices and replaying transactions…"):
                results = analyze_scenarios(trades, scenario_trades, bench, start, end, rate, initial_cash, currencies,
                                            cached_download=get_prices, clear_cache=get_prices.clear)
            original, variant, replay_error = results
            st.session_state.analysis = (signature, (original, variant, scenario_error or replay_error))
        except PortfolioError as exc:
            st.error(str(exc))
    previous = st.session_state.get("analysis")
    if previous is not None:
        previous_signature, results = previous
        if previous_signature == signature:
            show_results(*results, bench, rate, markers, shade, compare, logarithmic, edited)
        else:
            st.info("Inputs changed. Run analysis to update your results.")


if __name__ == "__main__":
    main()
