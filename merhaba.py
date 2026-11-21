from calendar import month_abbr
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analytics import PERCENT_METRICS, RISK_FREE_RATE, drawdown, monthly_returns, rolling_metrics, stats
from market_data import load_prices
from portfolio import (
    ACTIONS, COLUMNS, PortfolioError, QuoteError, backtest, export_log, import_log,
    parse_log, quoted_symbols, symbol,
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
    return pd.DataFrame(rows, columns=COLUMNS, dtype="string")


def replace_draft(frame, message):
    st.session_state.editor_seed = frame.reset_index(drop=True)
    st.session_state.editor_version += 1
    st.session_state.editor_message = message
    st.rerun()


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
            "**Sell** liquidates the named positions. **Sell & Buy** spends percentages of the sale proceeds. "
            "Enter tickers and percentages in matching comma-separated lists. Blank weights split the full budget equally."
        )
    edited = st.data_editor(
        state.editor_seed, key=f"log_{state.editor_version}", num_rows="dynamic",
        hide_index=True, width="stretch",
        column_config={
            "Date": st.column_config.TextColumn("Date", help="YYYY-MM-DD or YYYYMMDD"),
            "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS),
            "Sell": st.column_config.TextColumn("Sell tickers"),
            "Buy": st.column_config.TextColumn("Buy tickers"),
            "Weights": st.column_config.TextColumn("Weights %", help="In the same order as the buy tickers"),
        },
    )
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


def growth_chart(result, benchmark, markers, shade):
    ledger = result.ledger
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    ax = fig.subplots()
    ax.plot(ledger.index, ledger["Portfolio"], label="Portfolio", color=BLUE, linewidth=2.4)
    ax.plot(ledger.index, ledger["Benchmark"], label=benchmark, color=ORANGE, linestyle="--", linewidth=1.6)
    if len(ledger) == 1:
        ax.scatter(ledger.index, ledger["Portfolio"], color=BLUE)
    if shade:
        declines, peak, trough = drawdown(ledger["Portfolio"])
        if declines.min() < 0:
            ax.axvspan(peak, trough, color="#bf6870", alpha=0.18, label="Max drawdown")
    if markers and not result.executions.empty:
        for day, group in result.executions.groupby("Executed", sort=False):
            ax.axvline(day, color="#66757b", linewidth=0.8, linestyle=":", alpha=0.65)
            if len(result.executions) <= 10:
                ax.annotate(" / ".join(group["Action"]), xy=(day, 0.97),
                            xycoords=("data", "axes fraction"), xytext=(3, 0),
                            textcoords="offset points", rotation=90, va="top", fontsize=8)
    ax.set_ylabel("Value per 100 invested")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", frameon=False)
    st.pyplot(fig)


def formatted(name, value):
    if not np.isfinite(value):
        return "—"
    return f"{value:.2%}" if name in PERCENT_METRICS else f"{value:,.2f}"


def risk_charts(result, bench, compare_rolling):
    portfolio = result.ledger["Portfolio"]
    returns = portfolio.pct_change(fill_method=None).dropna()
    if returns.empty:
        st.info("A second closing valuation is needed for return distributions.")
        return
    with st.expander("Monthly returns"):
        months = monthly_returns(portfolio)
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


def show_results(result, bench, cash_rate, markers, shade, compare_rolling, draft):
    ledger = result.ledger
    first, last = ledger.index[0].date(), ledger.index[-1].date()
    summary = pd.DataFrame({name: stats(ledger[name]) for name in ("Portfolio", "Benchmark")})
    port, reference = summary["Portfolio"], summary["Benchmark"]
    spread = port["Total return"] - reference["Total return"]
    st.divider()
    st.header("Performance")
    st.caption(f"{first:%d %b %Y} – {last:%d %b %Y} · {len(ledger):,} benchmark closes · Starting value of 100 for both series")
    top = st.columns(4)
    top[0].metric("Portfolio return", formatted("Total return", port["Total return"]))
    top[1].metric(f"{bench} return", formatted("Total return", reference["Total return"]))
    top[2].metric("Return difference", f"{spread*100:+.2f} pp")
    top[3].metric("Max drawdown", formatted("Max drawdown", port["Max drawdown"]))
    growth_chart(result, bench, markers, shade)
    if result.pending:
        st.info("Rows outside the available closing dates: " + ", ".join(map(str, result.pending)) + ".")
    if result.carried:
        details = ", ".join(f"{ticker} ({n})" for ticker, n in sorted(result.carried.items()))
        st.warning("Held positions valued using a previous close on these sessions: " + details + ".")
    with st.expander("Holdings and transaction history", expanded=True):
        allocations = result.positions.iloc[-1]
        holdings = [(ticker, value) for ticker, value in allocations.items() if value > 0]
        holdings.append(("Cash", ledger["Cash"].iloc[-1]))
        allocation = pd.DataFrame(holdings, columns=["Position", "Closing value"])
        allocation["Weight %"] = allocation["Closing value"] / ledger["Portfolio"].iloc[-1] * 100
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
        cash_weight = ledger["Cash"] / ledger["Portfolio"]
        st.caption(f"Average session-end allocation: {1-cash_weight.mean():.1%} invested and {cash_weight.mean():.1%} cash.")
    pretty = pd.DataFrame({name: [formatted(metric, v) for metric, v in col.items()]
                           for name, col in summary.items()}, index=summary.index).rename(columns={"Benchmark": bench + " (benchmark)"})
    with st.expander("Detailed statistics"):
        st.dataframe(pretty)
        for name, title in (("Portfolio", "Portfolio"), ("Benchmark", bench)):
            values = ledger[name]
            declines, peak, trough = drawdown(values)
            if declines.min() < 0:
                st.caption(f"{title} drawdown peak {peak.date()} · trough {trough.date()} · {(trough - peak).days} calendar days")
            changes = values.pct_change(fill_method=None).dropna()
            if not changes.empty:
                st.caption(f"{title} best day {changes.idxmax().date()} · worst day {changes.idxmin().date()}")
        st.caption(f"Risk metrics use 252 sessions per year and a {RISK_FREE_RATE:.0%} annual risk-free assumption. A dash indicates an undefined metric.")
    risk_charts(result, bench, compare_rolling)
    with st.expander("Download results"):
        out = ledger.rename(columns=PERFORMANCE_NAMES).copy()
        out["Portfolio Daily Return"] = ledger["Portfolio"].pct_change(fill_method=None)
        out["Benchmark Daily Return"] = ledger["Benchmark"].pct_change(fill_method=None)
        out = out.join(result.positions.add_prefix("Position value: "))
        report = "\n".join([
            "PORTFOLIO VISUALIZER", f"{first} to {last}", f"Benchmark: {bench}",
            f"Cash rate: {cash_rate:.2f}% per year, compounded each benchmark session.",
            "Values are per 100 invested at the first reported close.",
            f"Portfolio minus benchmark: {spread * 100:+.2f} percentage points.",
            f"Average cash allocation: {cash_weight.mean():.2%}", "", pretty.to_string(),
            "", "Executed transactions in the reporting window",
            result.executions.to_string(index=False) if not result.executions.empty else "None in the window.",
            "", "Unexecuted row numbers: " + (", ".join(map(str, result.pending)) or "None"),
            "Carried-forward marks (ticker and number of sessions): " + (str(result.carried) if result.carried else "None"),
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
    st.caption("Replay your transactions and compare the growth of 100 against a benchmark.")
    with st.sidebar:
        st.header("Analysis settings")
        start = st.date_input("Start date", date(2013, 1, 1), min_value=date(1900, 1, 1),
                              max_value=date.today(), key="start_date")
        end = st.date_input("End date", date.today(), min_value=date(1900, 1, 1),
                            max_value=date.today(), key="end_date")
        bench = symbol(st.text_input("Benchmark ticker", "SPY", key="benchmark"))
        rate = st.number_input("Annual cash interest (%)", min_value=0.0, max_value=100.0,
                               value=4.5, step=0.25, key="cash_rate")
        st.subheader("Chart settings")
        markers = st.checkbox("Mark trades", True)
        shade = st.checkbox("Highlight max drawdown", False)
        compare = st.checkbox("Show benchmark on rolling charts", True)
        with st.expander("Valuation assumptions"):
            st.write("Trades execute at the next benchmark close on or after the date entered. "
                     "Earlier holdings earn the move into that close. Same-date trades follow log order. "
                     "Prices include split and dividend adjustments. Cash compounds once per benchmark session, "
                     "using 252 sessions per year. Use assets and a benchmark in one currency. "
                     "The model uses fractional positions and assumes zero fees and taxes.")
    edited = editor()
    error = None
    try:
        trades = parse_log(edited)
        if not bench or "," in bench or any(c.isspace() for c in bench):
            raise PortfolioError("Enter one benchmark ticker.")
        if start is None or end is None or start > end:
            raise PortfolioError("Choose an end date on or after the start date.")
        if trades[0].date > pd.Timestamp(end):
            raise PortfolioError("The first transaction is after the end date.")
    except PortfolioError as exc:
        error = str(exc)
        st.warning(error)
    signature = (export_log(edited), start, end, bench, rate)
    if st.button("Run analysis", key="run", type="primary", disabled=error is not None):
        st.session_state.pop("analysis", None)
        tickers = quoted_symbols(trades, bench, end)
        # A prior quote distinguishes an ordinary opening holiday from a younger benchmark.
        fetch_from = (trades[0].date - pd.Timedelta(days=30)).date()
        try:
            with st.spinner("Loading adjusted prices and replaying transactions…"):
                prices = get_prices(tickers, fetch_from, end)
                result = backtest(prices, trades, bench, start, end, rate)
            st.session_state.analysis = (signature, result)
        except QuoteError as exc:
            get_prices.clear(tickers, fetch_from, end)
            st.error(str(exc))
        except PortfolioError as exc:
            st.error(str(exc))
    previous = st.session_state.get("analysis")
    if previous is not None:
        previous_signature, result = previous
        if previous_signature == signature:
            show_results(result, bench, rate, markers, shade, compare, edited)
        else:
            st.info("Inputs changed. Run analysis to update your results.")


if __name__ == "__main__":
    main()
