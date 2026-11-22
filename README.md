# Portfolio Visualizer

Replay a dated transaction log and see how your portfolio performed alongside your original holdings and a market benchmark. The comparison shows the ending value of 100 and the maximum drawdown for each strategy. The app also includes a daily holdings and cash ledger, transaction history, risk charts, and CSV and text downloads.

## Run locally

Open a terminal in the repository root. These commands use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run merhaba.py
```

Use `python3` in the first command if it already points to Python 3.12. Streamlit opens the app in your browser. Select **Run analysis** to try the preloaded tech example, or use **Import or export transaction CSV** to load your previous transaction export. Yahoo Finance price downloads require internet access.

## Use your existing transaction CSV

The editor accepts the five columns from the original app, plus an optional `Sell %` column. Field order in the CSV can vary. Quote fields containing commas. When all sell percentages are blank, the download uses the original five-column format.

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

Blank weights split the available budget equally. A total below 100% retains the remainder in cash. An Initial or Rebalance row with blank tickers and weights holds everything in cash. Transactions on the same requested date execute in their input order.

In the example, the second row sells 25% of the MSFT shares held at execution and spends 60% of those proceeds on GOOGL. To sell different fractions of two positions, enter `AAPL,MSFT` under Sell and `25,100` under Sell %. Each percentage applies to that ticker's current shares.

## How the backtest works

- The first eligible transaction funds the model with 100. The benchmark's quoted sessions supply the calendar. A weekend or holiday instruction executes at the next available benchmark close. The positions held at the previous close earn the price change into the execution close.
- The ledger tracks fractional units using Yahoo's split- and dividend-adjusted closes. These units model reinvested total returns. Between trades, portfolio weights drift with market values. The transactions use one currency and assume zero commissions, slippage and taxes.
- Cash earns the annual effective rate entered in the sidebar, compounded once for each benchmark session after the funding close. A sale at today's close starts earning interest over the next session.
- Buy and hold keeps the shares and cash remaining immediately after the first transaction, including when several rows execute at the same close. Its cash earns the selected rate. The engine replays transactions before the selected reporting start, then scales each strategy independently to 100 at the first reported close. Ending values and drawdowns refer to that window. Holdings and transaction amounts use the active portfolio's scale.
- Trades require fresh prices for every position being bought or sold. When a needed price is missing, the app requests the affected symbols once more and keeps prices already received. If the retry still cannot price a trade, the error identifies the missing data. On a valuation day, an existing holding can use its last available close. The app lists affected symbols and session counts for each strategy.
- Transactions that place no orders appear next to the performance summary with the log row and reason. This includes earlier rows replayed to establish opening holdings. The daily CSV includes all three value series, and the text report records the zero-order rows.
- The end date is inclusive. Rows scheduled beyond the last available close stay pending. The first daily return is blank because that row establishes the starting balance.

## Check the installation

```bash
python -m unittest discover -s tests -v
```

The offline tests cover portfolio and buy-and-hold accounting, partial sales, weekend trades, price retries, cash interest, benchmark alignment and the Streamlit CSV workflow.

## Streamlit Community Cloud

This project runs from `merhaba.py` on the `main` branch. The supporting Python modules and `requirements.txt` live alongside it. [Open the portfolio app](https://stock-app-drkienpnjsbecpxcxo3ina.streamlit.app)
