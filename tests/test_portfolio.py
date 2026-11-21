import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analytics import drawdown, monthly_returns, stats
from market_data import load_prices
from portfolio import COLUMNS, PortfolioError, backtest, export_log, import_log, parse_log, quoted_symbols


def log(rows):
    return parse_log(pd.DataFrame(rows, columns=COLUMNS))


def simulate(data, rows, dates=None, start=None, end=None, rate=0.0):
    dates = (pd.bdate_range("2024-01-08", periods=len(next(iter(data.values()))))
             if dates is None else pd.to_datetime(dates))
    prices = pd.DataFrame(data, index=dates)
    if "SPY" not in prices:
        prices["SPY"] = 100.0
    return backtest(prices, log(rows), "SPY", start or dates[0], end or dates[-1], rate)


class LedgerTests(unittest.TestCase):
    def test_two_stocks_and_cash_earn_eight_percent(self):
        r = simulate({"A": [100, 110], "B": [100, 110]},
                     [("2024-01-08", "Initial", "", "A,B", "40,40")])
        np.testing.assert_allclose(r.ledger.Portfolio, [100, 108])
        np.testing.assert_allclose(r.ledger.Cash, [20, 20])
        np.testing.assert_allclose(r.positions.sum(axis=1) + r.ledger.Cash, r.ledger.Portfolio)

    def test_weights_drift_and_round_trip_returns_to_one_hundred(self):
        r = simulate({"A": [100, 200, 100], "B": [100, 100, 100]},
                     [("2024-01-08", "Initial", "", "A,B", "50,50")])
        np.testing.assert_allclose(r.ledger.Portfolio, [100, 150, 100])
        self.assertAlmostEqual(r.positions.A.iloc[1] / r.ledger.Portfolio.iloc[1], 2/3)

    def test_explicit_rebalance_really_can_earn_twelve_and_a_half_percent(self):
        r = simulate({"A": [100, 200, 100], "B": [100, 100, 100]}, [
            ("2024-01-08", "Initial", "", "A,B", "50,50"),
            ("2024-01-09", "Rebalance", "", "A,B", "50,50"),
        ])
        self.assertAlmostEqual(r.ledger.Portfolio.iloc[-1], 112.5)

    def test_weekend_trade_keeps_friday_to_monday_return(self):
        r = simulate({"SPY": [100, 100, 110, 110]}, [
            ("2024-01-04", "Initial", "", "SPY", "100"),
            ("2024-01-07", "Rebalance", "", "SPY", "100"),
        ], ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
        np.testing.assert_allclose(r.ledger.Portfolio, r.ledger.Benchmark)
        self.assertAlmostEqual(r.ledger.Portfolio.iloc[-1], 110)
        self.assertEqual(r.executions.Executed.iloc[-1], pd.Timestamp("2024-01-08"))

    def test_weekend_switch_uses_outgoing_holding_for_overnight_move(self):
        r = simulate({"A": [100, 110, 110], "B": [100, 50, 55]}, [
            ("2024-01-05", "Initial", "", "A", ""),
            ("2024-01-07", "Sell & Buy", "A", "B", ""),
        ], ["2024-01-05", "2024-01-08", "2024-01-09"])
        np.testing.assert_allclose(r.ledger.Portfolio, [100, 110, 121])

    def test_benchmark_discards_gains_before_funding(self):
        r = simulate({"SPY": [100, 105, 120, 120, 120]},
                     [("2024-01-10", "Initial", "", "SPY", "100")])
        np.testing.assert_allclose(r.ledger[["Portfolio", "Benchmark"]], 100)
        self.assertEqual(r.ledger.index[0], pd.Timestamp("2024-01-10"))

    def test_younger_benchmark_cannot_silently_delay_old_trades(self):
        with self.assertRaisesRegex(PortfolioError, "predate the available SPY history"):
            simulate({"A": [100, 200, 200], "SPY": [np.nan, np.nan, 100]},
                     [("2024-01-08", "Initial", "", "A", "100")])
        r = simulate({"A": [100, 110, 150, 170], "SPY": [100, np.nan, 102, 105]},
                     [("2024-01-09", "Initial", "", "A", "100")])
        self.assertEqual(r.executions.Executed.iloc[0], pd.Timestamp("2024-01-10"))

    def test_same_close_order_and_drifted_sale_proceeds(self):
        r = simulate({"A": [100, 200, 200], "B": [100]*3, "C": [100]*3}, [
            ("2024-01-08", "Initial", "", "A", "50"),
            ("2024-01-09", "Sell & Buy", "A", "B", "60"),
            ("2024-01-09", "Buy", "", "C", "50"),
        ])
        snapshots = r.executions
        self.assertEqual(snapshots["Log row"].tolist(), [1, 2, 3])
        np.testing.assert_allclose(snapshots.Sold, [0, 100, 0])
        np.testing.assert_allclose(snapshots.Bought, [50, 60, 45])
        np.testing.assert_allclose(snapshots["Cash after"], [50, 90, 45])
        np.testing.assert_allclose(r.positions.iloc[-1][["A", "B", "C"]], [0, 60, 45])
        self.assertAlmostEqual(r.ledger.Portfolio.iloc[-1], 150)

    def test_all_cash_interval_accrues_interest_on_every_close(self):
        r = simulate({"A": [100]*8}, [
            ("2024-01-08", "Initial", "", "A", "100"),
            ("2024-01-10", "Sell", "A", "", ""),
            ("2024-01-15", "Buy", "", "A", "100"),
        ], rate=4.5)
        self.assertEqual(len(r.ledger), 8)
        self.assertAlmostEqual(r.ledger.Portfolio.iloc[-1], 100 * 1.045**(3/252))
        self.assertAlmostEqual(r.ledger.Cash.iloc[-1], 0)

    def test_entirely_cash_funded_at_first_close(self):
        r = simulate({"SPY": [100, 99, 102]}, [("2024-01-08", "Initial", "", "", "")], rate=5)
        self.assertEqual(len(r.positions.columns), 0)
        self.assertAlmostEqual(r.ledger.Cash.iloc[-1], 100 * 1.05**(2/252))
        np.testing.assert_allclose(r.ledger.Portfolio, r.ledger.Cash)

    def test_replay_before_reporting_window_and_rebase_both_curves(self):
        kwargs = dict(data={"A": [100, 200, 300, 250, 300], "SPY": [10, 20, 50, 55, 60]},
                      rows=[("2024-01-08", "Initial", "", "A", "50"),
                            ("2024-01-11", "Sell", "A", "", "")])
        full = simulate(**kwargs)
        sliced = simulate(**kwargs, start="2024-01-10")
        same = full.ledger.loc[sliced.ledger.index]
        np.testing.assert_allclose(sliced.ledger.Portfolio, 100*same.Portfolio/same.Portfolio.iloc[0])
        np.testing.assert_allclose(sliced.ledger.Benchmark, 100*same.Benchmark/same.Benchmark.iloc[0])
        self.assertAlmostEqual(sliced.positions.A.iloc[0], 75)
        self.assertEqual(sliced.executions.Action.tolist(), ["Sell"])
        self.assertAlmostEqual(sliced.executions["Cash after"].iloc[-1], sliced.ledger.Cash.iloc[-1])

    def test_future_transactions_never_run_or_enter_price_request(self):
        rows = [("2024-01-08", "Initial", "", "SPY", "100"),
                ("2024-01-13", "Buy", "", "UNKNOWN", "10")]
        r = simulate({"SPY": [100]*5}, rows, end="2024-01-13")
        self.assertEqual(r.pending, (2,))
        self.assertEqual(quoted_symbols(log(rows), "SPY", "2024-01-12"), ("SPY",))

    def test_prior_close_used_only_to_value_an_existing_position(self):
        r = simulate({"A": [100, np.nan, 120]}, [("2024-01-08", "Initial", "", "A", "100")])
        self.assertEqual(r.carried, {"A": 1})
        np.testing.assert_allclose(r.ledger.Portfolio, [100, 100, 120])
        with self.assertRaisesRegex(PortfolioError, "no fresh adjusted close for A"):
            simulate({"A": [100, np.nan, 120]}, [
                ("2024-01-08", "Initial", "", "A", "100"),
                ("2024-01-09", "Sell", "A", "", ""),
            ])
        with self.assertRaisesRegex(PortfolioError, "no fresh adjusted close"):
            simulate({"A": [np.nan, 100]}, [("2024-01-08", "Initial", "", "A", "100")])

    def test_free_cash_and_invalid_sells(self):
        data = {"A": [100, 110], "B": [100, 100]}
        first = ("2024-01-08", "Initial", "", "A", "100")
        r = simulate(data, [first, ("2024-01-09", "Buy", "", "B", "100")])
        self.assertEqual(r.executions.Note.iloc[-1], "No cash available")
        self.assertEqual(r.executions.Bought.iloc[-1], 0)
        with self.assertRaisesRegex(PortfolioError, "unheld position"):
            simulate(data, [first, ("2024-01-09", "Sell", "B", "", "")])

    def test_explicit_zero_weight_and_cash_rebalance(self):
        r = simulate({"A": [100, 110, 90]}, [
            ("2024-01-08", "Initial", "", "A", "60"),
            ("2024-01-09", "Buy", "", "A", "0"),
            ("2024-01-09", "Rebalance", "", "", ""),
        ])
        self.assertAlmostEqual(r.ledger.Cash.iloc[-1], 106)
        self.assertAlmostEqual(r.ledger.Portfolio.iloc[-1], 106)


class InputAndMetricsTests(unittest.TestCase):
    def test_old_csv_round_trip_bom_quoted_fields_blank_cells_and_ticker_na(self):
        source = ('Date,Action,Sell,Buy,Weights\r\n'
                  '20240108,Initial,,"BRK.B,NA","25, 25"\r\n'
                  '2024-01-09,Buy,,GOOGL,\r\n').encode("utf-8-sig")
        frame = import_log(source)
        pd.testing.assert_frame_equal(frame, import_log(export_log(frame)))
        first, second = parse_log(frame)
        self.assertEqual(first.buys, ("BRK-B", "NA"))
        self.assertEqual(second.fractions, (1.0,))

    def test_invalid_rows_dates_weights_and_unquoted_csv(self):
        for weights in ["20,90", "-1,50", "nan,20", "inf,0", "40", "abc,50"]:
            with self.subTest(weights=weights), self.assertRaisesRegex(PortfolioError, "Row 1"):
                log([("2024-01-08", "Initial", "", "A,B", weights)])
        for row in [("", "Buy", "", "A", ""), ("2024-02-30", "Buy", "", "A", ""),
                    ("2024-01-08", "Deposit", "", "A", ""), ("2024-01-08", "Buy", "", "A,A", "")]:
            with self.subTest(row=row), self.assertRaises(PortfolioError):
                log([row])
        with self.assertRaisesRegex(PortfolioError, "CSV line 2"):
            import_log(b"Date,Action,Sell,Buy,Weights\n2024-01-08,Initial,,A,B,50,50")
        with self.assertRaisesRegex(PortfolioError, "needs these columns"):
            import_log(b"Date,Portfolio Value\n2024-01-08,100")

    def test_actual_months_and_opening_drawdown(self):
        series = pd.Series([100., 90., 99.], index=pd.to_datetime(["2024-01-29", "2024-01-31", "2024-02-01"]))
        np.testing.assert_allclose(monthly_returns(series), [-0.1, 0.1])
        declines, peak, trough = drawdown(series)
        self.assertAlmostEqual(declines.min(), -0.1)
        self.assertEqual((peak, trough), (series.index[0], series.index[1]))
        self.assertTrue(np.isnan(stats(series.iloc[:1])["CAGR"]))
        self.assertAlmostEqual(stats(series)["CAGR"], .99**(365.25/3)-1)

    @patch("market_data.yf.download")
    def test_adjusted_prices_request_includes_end_and_matches_symbols(self, download):
        dates = pd.to_datetime(["2024-01-08", "2024-01-09"])
        close = pd.DataFrame({"B": [5, 6], "A": [3, 4]}, index=dates)
        download.return_value = pd.concat({"Close": close}, axis=1)
        got = load_prices(("A", "B"), dates[0], dates[-1])
        self.assertEqual(got.columns.tolist(), ["A", "B"])
        self.assertEqual(download.call_args.kwargs["end"], "2024-01-10")
        self.assertIs(download.call_args.kwargs["auto_adjust"], True)
        download.return_value = pd.DataFrame({"Close": [4, 5]}, index=dates)
        self.assertEqual(load_prices(("A",), dates[0], dates[-1]).A.tolist(), [4, 5])
        download.return_value = pd.DataFrame()
        with self.assertRaisesRegex(PortfolioError, "no prices"):
            load_prices(("A",), dates[0], dates[-1])


if __name__ == "__main__":
    unittest.main()
