"""Interactive performance chart and close-by-close transaction details."""

import altair as alt
import pandas as pd

from analytics import drawdown
from portfolio import BUY_AND_HOLD


def trade_markers(result, strategy):
    rows = []
    for day, events in result.executions.groupby("Executed", sort=False):
        details = []
        for _, row in events.iterrows():
            parts = [f"Row {row['Log row']} {row['Action']}"]
            if row["Sell order"]:
                parts.append(f"sell {row['Sell order']} ({row['Account sold']:,.2f})")
            if row["Buy tickers"]:
                parts.append(f"buy {row['Buy tickers']} ({row['Account bought']:,.2f})")
            if row["External flow"]:
                parts.append(f"flow {row['External flow']:+,.2f}")
            if row["Fee"]:
                parts.append(f"fee {row['Fee']:,.2f}")
            if row["Note"]:
                parts.append(row["Note"])
            parts.append(f"cash {row['Account cash after']:,.2f}")
            details.append(" · ".join(parts))
        rows.append({"Date": day, "Value": result.ledger.at[day, "Portfolio"],
                     "Strategy": strategy, "Transactions": " | ".join(details)})
    return pd.DataFrame(rows)


def growth_chart(result, alternative, *, markers=True, shade=False, logarithmic=False):
    frame = result.ledger[["Portfolio", BUY_AND_HOLD, "Benchmark"]].copy()
    colors = ["#176a78", "#7451a4", "#a75020"]
    dashes = [[1, 0], [6, 3], [3, 3]]
    if alternative is not None:
        frame["Scenario"] = alternative.ledger["Portfolio"]
        colors.append("#0079c4")
        dashes.append([9, 2, 2, 2])
    names = frame.columns.tolist()
    color = alt.Color("Strategy:N", scale=alt.Scale(domain=names, range=colors),
                      legend=alt.Legend(title=None, orient="bottom"))
    y = alt.Y("Value:Q", title="Performance per 100", scale=alt.Scale(type="log" if logarithmic else "linear", zero=False))
    x = alt.X("Date:T", title=None)
    lines = alt.Chart(frame.reset_index()).transform_fold(names, as_=["Strategy", "Value"]).mark_line(
        point=len(frame) == 1, strokeWidth=2.2, clip=True
    ).encode(x=x, y=y, color=color,
             strokeDash=alt.StrokeDash("Strategy:N", scale=alt.Scale(domain=names, range=dashes), legend=None),
             tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), "Strategy:N", alt.Tooltip("Value:Q", format=",.2f")])
    layers = []
    if shade:
        declines, peak, trough = drawdown(frame["Portfolio"], opening=100)
        if declines.min() < 0:
            span = pd.DataFrame({"Start": [peak], "End": [trough]})
            layers.append(alt.Chart(span).mark_rect(color="#bf6870", opacity=0.16).encode(x="Start:T", x2="End:T"))
    layers.append(lines)
    if markers:
        for label, run, offset in (("Portfolio", result, -4 if alternative is not None else 0),
                                   ("Scenario", alternative, 4)):
            if run is None or run.executions.empty:
                continue
            marker = alt.Chart(trade_markers(run, label)).mark_point(filled=True, size=75, stroke="white", clip=True)
            layers.append(marker.encode(x=x, xOffset=alt.value(offset), y=y, color=color, tooltip=[
                alt.Tooltip("Date:T", format="%d %b %Y"), "Strategy:N",
                alt.Tooltip("Transactions:N", title=f"Trades and cash ({run.currency})"),
            ]))
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return alt.layer(*layers).properties(height=365).add_params(zoom)
