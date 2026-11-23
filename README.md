# Portfolio Visualizer

Replay a transaction log, change individual decisions in a separate scenario, and compare both with the original holdings and a benchmark. The app reports actual account money, performance per 100, maximum drawdown, and the contribution of each holding.

## Run locally

Open a terminal in the repository root. These commands use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run merhaba.py
```

Use `python3` in the first command if it already points to Python 3.12. Streamlit opens the app in your browser. Select **Run analysis** to try the preloaded tech example, or use **Import or export transaction CSV** to load your previous transaction export. Yahoo Finance price downloads require internet access.

## Compare a decision

Turn on **Compare a what-if scenario**. The app copies the current log. Check **Skip** on a scenario row or edit its date, tickers, weights, partial sale, cash flow or fee. In the Tech growth example, skip the 2017 MSFT-to-INTC switch to measure its effect with the 2022 rebalance still in place. **Copy current original into scenario** makes a fresh copy.

Choose **Run analysis**. Both versions use one quote snapshot, starting balance, funding date, benchmark calendar and currency settings. A sale of an unheld position identifies the scenario row and leaves the original results visible. A scenario with all rows skipped holds its starting cash. Switch the growth chart to a logarithmic scale in the sidebar or hover over trade markers for the orders and remaining cash.

Use **Save or open a comparison** to download both editable logs, skipped rows and financial settings as one JSON file. Open that file to resume in another session and run the analysis against available quotes. Daily CSVs and reports preserve results for a particular run.

## Use your existing transaction CSV

The original five columns still work. Optional columns are `Sell %`, `Amount`, and `Fee`. Downloads include the optional columns you used. Input column order can vary. Quote fields containing commas.

```csv
Date,Action,Sell,Buy,Weights,Sell %
2024-01-05,Initial,,"AAPL,MSFT","40,40",
2024-01-14,Sell & Buy,MSFT,GOOGL,60,25
2024-02-01,Buy,,AAPL,50,
```

| Action | Source of funds | Effect |
| --- | --- | --- |
| Initial | Starting balance | Establish positions at the specified percentages. |
| Buy | Available cash | Invest the indicated percentages of cash. |
| Sell | Named positions | Sell the indicated percentage of each position and retain the proceeds in cash. Blank sells each position in full. |
| Sell & Buy | Proceeds of this sale | Buy the new positions with the specified share of the proceeds. Earlier cash stays available. |
| Rebalance | Entire portfolio | Reallocate at the effective close, including any cash. |
| Deposit | External money | Add the positive Amount to cash. |
| Withdrawal | Available cash | Remove the positive Amount. An unfunded withdrawal identifies its log row. |
| Fee | Available cash | Deduct the positive Amount as a standalone expense. |

Blank weights split the available budget equally. A total below 100% retains the remainder in cash. An Initial or Rebalance row with blank tickers and weights holds everything in cash. Transactions on the same requested date execute in their input order.

In the example, the second row sells 25% of the MSFT shares held at execution and spends 60% of those proceeds on GOOGL. To sell different fractions of two positions, enter `AAPL,MSFT` under Sell and `25,100` under Sell %. Each percentage applies to that ticker's current shares.

## Money, returns and currencies

Set **Starting cash** to the actual opening balance. It defaults to 100. A zero opening balance can be funded with a Deposit row. Enter cash flows and flat fees in the **Account currency**. A trade's Fee is reserved before purchases. Sell & Buy spends the net proceeds of that sale. A fee must be covered by cash or sale proceeds and leave positive equity.

Account balance includes deposits and withdrawals. Investment gain is the balance less net contributed capital since funding. Performance per 100 uses fund units. External flows issue or redeem units at the valuation immediately before the flow. Fees reduce the unit value. A full withdrawal leaves the index at its last value until new capital arrives.

**Report in** sets the currency for both strategies and the benchmark. Changing it preserves the account currency, cash flows and underlying trades. Yahoo's `TRY=X` gives lira per US dollar. Istanbul symbols ending in `.IS` use TRY, and US symbols use USD. Set explicit overrides for other USD or TRY quotes. For portfolios quoted entirely in a different currency, **As quoted** preserves their native prices. FX uses the most recent observation on or before each close. An Istanbul market holiday can still change the value of a held position in USD.

**Return by holding and worst drawdown** reconciles each strategy to its total return and its own peak-to-trough drawdown. Contributions use the positions carried into each valuation. Cash includes interest and currency effects. Fees are shown separately. Both columns are in percentage points.

## How the backtest works

- The original log's first date sets the shared funding date. The benchmark's quoted sessions supply the calendar. A weekend or holiday instruction executes at the next available benchmark close. The positions held at the previous close earn the price change into the execution close.
- The ledger tracks fractional units using Yahoo's split- and dividend-adjusted closes. These units model reinvested total returns. Between trades, portfolio weights drift with market values. Trading uses the entered fees and assumes zero slippage and taxes.
- Cash earns the annual effective rate entered in the sidebar, compounded once for each benchmark session after the funding close. A sale at today's close starts earning interest over the next session.
- Buy and hold keeps the shares and cash remaining immediately after the first transaction, including when several rows execute at the same close. Its cash earns the selected rate. The passive portfolio tracks its original composition independently of subsequent external flows. When the window starts at funding, opening fees count against the initial 100. A window beginning later rebases each curve to its first reported close. The account tables retain actual amounts.
- Trades require fresh prices for every position being bought or sold. When a needed price is missing, the app requests the affected symbols once more and keeps prices already received. If the retry still cannot price a trade, the error identifies the missing data. On a valuation day, an existing holding can use its last available close. The app lists affected symbols and session counts for each strategy.
- Transactions that place no orders appear next to the performance summary with the log row and reason. This includes earlier rows replayed to establish opening holdings. The daily CSV includes the scenario when enabled, account money, and daily return contributions. The text report includes the comparison and the transaction history.
- The end date is inclusive. Rows scheduled beyond the last available close stay pending. The first daily return is blank unless an opening fee reduces the initial value.

## Check the installation

```bash
python -m unittest discover -s tests -v
```

The offline tests cover portfolio and buy-and-hold accounting, trade execution, price retries, cash flows, fees, currency conversion, attribution and the Streamlit CSV and saved-comparison workflows.

## Streamlit Community Cloud

This project runs from `merhaba.py` on the `main` branch. The supporting Python modules and `requirements.txt` live alongside it. [Open the portfolio app](https://stock-app-drkienpnjsbecpxcxo3ina.streamlit.app)
