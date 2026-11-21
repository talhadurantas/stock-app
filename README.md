# Portfolio Visualizer

Replay a dated portfolio transaction log and compare the growth of 100 against a benchmark. The app includes a daily holdings and cash ledger, a record of executed trades, a monthly return heatmap, rolling performance and risk metrics, and CSV and text downloads.

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

The editor accepts the five columns from the original app. Field order in the CSV can vary. Quote fields containing commas.

```csv
Date,Action,Sell,Buy,Weights
2024-01-05,Initial,,"AAPL,MSFT","40,40"
2024-01-14,Sell & Buy,MSFT,GOOGL,60
2024-02-01,Buy,,AAPL,50
```

| Action | Source of funds | Effect |
| --- | --- | --- |
| Initial | Starting balance | Establish positions at the specified percentages. |
| Buy | Available cash | Invest the indicated percentages of cash. |
| Sell | Named positions | Sell each named position in full and retain the proceeds in cash. |
| Sell & Buy | Proceeds of this sale | Buy the new positions with the specified share of the proceeds. Earlier cash stays available. |
| Rebalance | Entire portfolio | Reallocate at the effective close, including any cash. |

Blank weights split the available budget equally. A total below 100% retains the remainder in cash. An Initial or Rebalance row with blank tickers and weights holds everything in cash. Transactions on the same requested date execute in their input order.

## How the backtest works

- The first eligible transaction funds the model with 100. The benchmark's quoted sessions supply the calendar. A weekend or holiday instruction executes at the next available benchmark close. The positions held at the previous close earn the price change into the execution close.
- The ledger tracks fractional units using Yahoo's split- and dividend-adjusted closes. These units model reinvested total returns. Between trades, portfolio weights drift with market values. The transactions use one currency and assume zero commissions, slippage and taxes.
- Cash earns the annual effective rate entered in the sidebar, compounded once for each benchmark session after the funding close. A sale at today's close starts earning interest over the next session.
- The engine replays any transactions before the selected reporting start. Both plotted series start at 100 on the first reported close. All displayed monetary amounts use those same units. CAGR uses the elapsed calendar days between the first and last reported closes.
- Trades require fresh prices for every position being bought or sold. A missing quote for an existing holding uses the last available close for valuation. The app lists affected symbols and counts. Use a benchmark with history covering the transaction log and check the requested and executed dates in the history.
- The end date is inclusive. Rows scheduled beyond the last available close stay pending. The first daily return is blank because that row establishes the starting balance.

## Check the installation

```bash
python -m unittest discover -s tests -v
```

The offline tests include the 40/40/20 example, the 50/50 round trip, weekend trades, data gaps, cash interest, benchmark alignment and a Streamlit CSV import and export workflow.

## Streamlit Community Cloud

This project runs from `merhaba.py` on the `main` branch. The supporting Python modules and `requirements.txt` live alongside it. [Open the portfolio app](https://stock-app-drkienpnjsbecpxcxo3ina.streamlit.app)
