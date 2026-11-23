"""Streamlit performance views and their downloadable results."""

from calendar import month_abbr

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analytics import (PERCENT_METRICS, active_returns, benchmark_stats, daily_returns, drawdown,
                       holding_breakdown, monthly_returns, rolling_metrics, stats)
from charts import growth_chart
from transactions import BUY_AND_HOLD, export_log


BLUE, ORANGE = "#176a78", "#a75020"
PERFORMANCE_NAMES = {"Portfolio": "Portfolio Value", "Benchmark": "Benchmark Value", "Cash": "Cash Value"}


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


def risk_charts(result, bench, compare_rolling, risk_free_rate):
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
                series = rolling_metrics(result.ledger[name], risk_free_rate=risk_free_rate).iloc[252:]
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


def transaction_table(view):
    events = view.executions.drop(columns=["Sold", "Bought", "Cash after"]).rename(columns={
        "Account sold": "Sold", "Account bought": "Bought", "Account cash after": "Cash after",
    })
    events = events.assign(Requested=events["Requested"].dt.date, Executed=events["Executed"].dt.date)
    st.dataframe(events.round({col: 3 for col in events.select_dtypes("number")}), hide_index=True)


def show_results(result, alternative, scenario_error, bench, cash_rate, risk_free_rate, markers, shade, compare_rolling, logarithmic, draft, price_source):
    ledger = result.ledger
    first, last = ledger.index[0].date(), ledger.index[-1].date()
    if alternative is not None:
        ledger = ledger.assign(Scenario=alternative.ledger["Portfolio"])
    strategies = ["Portfolio", BUY_AND_HOLD, "Benchmark"]
    if alternative is not None:
        strategies.insert(1, "Scenario")
    benchmark_observed = ledger["Benchmark"].reindex(result.benchmark_closes)
    relative_opening = 100 if ledger.index[0] in result.benchmark_closes else None
    summary = pd.DataFrame({name: pd.concat([
        stats(ledger[name], opening=100, risk_free_rate=risk_free_rate),
        benchmark_stats(ledger[name], benchmark_observed, opening=relative_opening),
    ]) for name in strategies})
    summary.loc["Ending value per 100"] = ledger.iloc[-1].reindex(strategies)
    pretty = pd.DataFrame({name: [formatted(metric, v) for metric, v in col.items()]
                           for name, col in summary.items()}, index=summary.index)
    pretty = pretty.rename(columns={"Benchmark": bench + " (benchmark)"})
    port, reference = summary["Portfolio"], summary["Benchmark"]
    spread = port["Total return"] - reference["Total return"]
    st.divider()
    st.header("Performance")
    st.caption(f"{first:%d %b %Y} – {last:%d %b %Y} · {len(ledger):,} valuation sessions · {price_source}")
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
    carried_benchmark = len(ledger) - len(benchmark_observed)
    if carried_benchmark:
        st.caption(f"{bench} uses its prior native close on {carried_benchmark} valuation sessions.")
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
        st.markdown(f"**Executed transactions in the selected window ({result.currency})**")
        if result.executions.empty:
            st.caption("Positions carried in from earlier transactions.")
        else:
            transaction_table(result)
        if alternative is not None:
            st.markdown("**Scenario transactions**")
            if not alternative.executions.empty:
                transaction_table(alternative)
            if not alternative.no_orders.empty:
                show_no_orders(alternative.no_orders)
            if alternative.pending:
                st.info("Pending scenario rows: " + ", ".join(map(str, alternative.pending)))
            if alternative.carried:
                st.caption(f"Scenario positions using previous closes (sessions): {alternative.carried}")
        cash_weight = result.account["Cash"].div(result.account["Balance"].replace(0, np.nan))
        average_cash = cash_weight.mean()
        if np.isfinite(average_cash):
            st.caption(f"Average funded session: {1-average_cash:.1%} invested and {average_cash:.1%} cash.")
        else:
            st.caption("The account held no capital during this window.")
    with st.expander("Detailed statistics"):
        st.dataframe(pretty)
        st.caption("Tracking error is the annualized standard deviation of daily returns above the benchmark. "
                   "The information ratio divides annualized mean active return by tracking error. Both use common "
                   "endpoints with an observed benchmark close, in the reporting currency.")
        for name in strategies:
            title = bench if name == "Benchmark" else name
            values = ledger[name]
            declines, peak, trough = drawdown(values, opening=100)
            if declines.min() < 0:
                st.caption(f"{title} drawdown peak {peak.date()} · trough {trough.date()} · {(trough - peak).days} calendar days")
            changes = daily_returns(values, 100).dropna()
            if not changes.empty:
                st.caption(f"{title} best day {changes.idxmax().date()} · worst day {changes.idxmin().date()}")
        st.caption(f"Risk metrics use 252 sessions per year and a {risk_free_rate:.2%} annual risk-free assumption in {result.currency}. A dash indicates an undefined metric.")
    risk_charts(result, bench, compare_rolling, risk_free_rate)
    with st.expander("Download results"):
        out = ledger[list(PERFORMANCE_NAMES)].rename(columns=PERFORMANCE_NAMES).copy()
        out["Portfolio Daily Return"] = daily_returns(ledger["Portfolio"], 100)
        out["Active Daily Return"] = active_returns(ledger["Portfolio"], benchmark_observed, relative_opening)
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
            "PORTFOLIO VISUALIZER", f"{first} to {last}", f"Benchmark: {bench}", f"Price source: {price_source}",
            f"Cash rate: {cash_rate:.2f}% per year, compounded on account-market sessions.",
            f"Annual risk-free rate in {result.currency}: {risk_free_rate:.2%}.",
            f"Account amounts in {result.currency}. Performance per 100 is adjusted for external cash flows.",
            pd.DataFrame(account_rows).to_string(index=False),
            *(label + " holding contributions (pp)\n" + holding_breakdown(view).to_string() for label, view in run_views.items()),
            *(["Scenario execution history\n" + alternative.executions.to_string(index=False)] if alternative is not None else []),
            *(["Scenario error: " + scenario_error] if scenario_error else []),
            "Buy and hold keeps the positions and cash left by the first transaction.",
            f"Portfolio minus benchmark: {spread * 100:+.2f} percentage points.",
            "Average cash allocation: " + (f"{average_cash:.2%}" if np.isfinite(average_cash) else "No funded sessions"), "", pretty.to_string(),
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
