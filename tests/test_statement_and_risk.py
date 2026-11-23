import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analytics import active_returns, benchmark_stats, rolling_metrics, stats
from projects import duplicate_log, read_project, save_project
from statements import read_balances, read_closes, read_fills, reconcile, replay_statement
from transactions import PortfolioError, import_log


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "statement"


def contents(name):
    return (EXAMPLE / name).read_bytes()


class StatementTests(unittest.TestCase):
    def setUp(self):
        self.fills = read_fills(contents("fills.csv"), "USD")
        self.closes = read_closes(contents("closes.csv"), self.fills, "USD")
        self.end, self.expected = read_balances(contents("balances.csv"))

    def test_reconciles_exact_fills_and_independent_statement_and_skip(self):
        original = replay_statement(self.fills,self.closes,0,self.end)
        alt = replay_statement(self.fills,self.closes,0,self.end,skipped=[5])
        self.assertTrue(reconcile(original,self.expected).Match.all())
        self.assertEqual((original.cash, original.daily.Balance.iloc[-1]), (679,1067.5))
        self.assertEqual((alt.cash, alt.daily.Balance.iloc[-1]), (460,1088.5))
        self.assertEqual(original.daily.Fees.sum(),3)
        self.assertEqual(alt.daily.Fees.sum(),2)
        self.assertEqual(original.daily.loc["2024-07-04","Balance"], 1007.1)
        del self.expected["Holding","DEMO"]
        self.expected["Cash",""] += .01
        self.assertEqual(reconcile(original,self.expected).Match.tolist(),[False,False,True])

    def test_dependent_sale_is_visible_and_malformed_inputs_rejected(self):
        with self.assertRaisesRegex(PortfolioError, "line 5.*insufficient cash or shares"):
            replay_statement(self.fills,self.closes,0,self.end,skipped=[3])
        with self.assertRaisesRegex(PortfolioError,"Missing unadjusted close.*DEMO.IS"):
            replay_statement(self.fills,self.closes.drop(index="2024-07-04"),0,self.end)
        for source in (contents("fills.csv").replace(b"-04:00,",b","),
                       contents("fills.csv").replace(b",USD,1,1,",b",USD,3,1,"),
                       contents("fills.csv").replace(b",300,TRY,",b",nan,TRY,")):
            with self.assertRaises(PortfolioError):
                read_fills(source,"USD")
        with self.assertRaisesRegex(PortfolioError,"twice"):
            read_balances(contents("balances.csv")+b"2024-07-16,Holding,DEMO,2\n")

    def test_cash_flows_and_split_keep_performance_flat(self):
        fills = read_fills(b"Timestamp,Action,Ticker,Quantity,Price,Currency,FX,Fee,Amount\n"
                           b"2024-07-02T10:00:00Z,Deposit,,,,,,,200\n"
                           b"2024-07-02T11:00:00Z,Buy,A,2,100,USD,1,,\n"
                           b"2024-07-03T09:00:00Z,Split,A,2,,,,,\n"
                           b"2024-07-03T11:00:00Z,Deposit,,,,,,,50\n"
                           b"2024-07-03T13:00:00Z,Withdrawal,,,,,,,40\n","USD")
        closes = read_closes(b"Date,Ticker,Close,Currency,FX\n2024-07-02,A,100,USD,1\n2024-07-03,A,50,USD,1\n", fills,"USD")
        r = replay_statement(fills,closes,0,pd.Timestamp("2024-07-03"))
        self.assertEqual((r.holdings,r.cash),({"A":4},10))
        np.testing.assert_allclose(r.daily["Performance per 100"],100)

    def test_demo_and_actual_upload_run_without_yahoo(self):
        with patch("market_data.yf.download",side_effect=AssertionError("offline mode reached Yahoo")) as yahoo:
            app = AppTest.from_file(str(ROOT/"merhaba.py"),default_timeout=30).run()
            app.button(key="offline_demo").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any("match the closing statement" in v.value for v in app.success))
            self.assertEqual(app.metric[0].value,"+21.00 USD")
            app.radio(key="statement_source").set_value("Upload my CSVs").run()
            for name in ("fills.csv","closes.csv","balances.csv"):
                data = contents(name)
                if name=="balances.csv":
                    data = data.replace(b"679.00",b"700.00")
                app.file_uploader(key="statement_"+name).upload(name,data,"text/csv").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("differs" in w.value for w in app.warning))
            yahoo.assert_not_called()


class RiskTests(unittest.TestCase):
    def test_ratios_use_the_supplied_hurdle_for_summary_and_rolling(self):
        days = pd.bdate_range("2022-01-03", periods=253)
        returns = np.tile([.01,-.005],126)
        values = pd.Series(100*np.cumprod(np.r_[1,1+returns]),index=days)
        for rate in (0.,.045,.32,-.01):
            excess = returns - ((1+rate)**(1/252)-1)
            sharpe = excess.mean()*np.sqrt(252)/returns.std(ddof=1)
            downside = np.sqrt(np.minimum(excess,0).dot(np.minimum(excess,0))/len(excess))
            summary = stats(values,risk_free_rate=rate)
            self.assertAlmostEqual(summary.Sharpe,sharpe)
            self.assertAlmostEqual(summary.Sortino,excess.mean()*np.sqrt(252)/downside)
            self.assertAlmostEqual(rolling_metrics(values,risk_free_rate=rate).Sharpe.iloc[-1],sharpe)

    def test_saved_hurdle_uses_the_reporting_currency_and_old_files_open(self):
        original = import_log(b"Date,Action,Sell,Buy,Weights\n2024-07-02,Initial,,US,100\n")
        config = dict(start_date=date(2024,7,2), end_date=date(2024,7,5), benchmark="SPY",
                      cash_rate=4.5, starting_cash=100., account_currency="USD",
                      report_currency="TRY", currency_overrides="")
        app = AppTest.from_file(str(ROOT/"merhaba.py"),default_timeout=30).run()
        for name, settings, expected in (("old",config,2.),("new",dict(config,risk_free_rate=41.5),41.5)):
            data = save_project(original, duplicate_log(original), settings)
            self.assertEqual(read_project(data)[2],settings)
            app.file_uploader(key="project_file").upload(name+".json",data,"application/json").run()
            app.button(key="open_project").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.number_input(key="hurdle_TRY").value,expected)
            self.assertEqual(app.number_input(key="cash_rate").value,4.5)

    def test_price_upload_and_reporting_hurdle_reach_results_without_a_network_call(self):
        original = import_log(b"Date,Action,Sell,Buy,Weights\n2024-07-02,Initial,,US,100\n")
        settings = dict(start_date=date(2024,7,2), end_date=date(2024,7,5), benchmark="SPY",
                        cash_rate=4.5, starting_cash=100., account_currency="USD", report_currency="TRY",
                        currency_overrides="", risk_free_rate=32., price_source="Adjusted-close CSV")
        data = save_project(original, duplicate_log(original), settings)
        dates = ["2024-07-02","2024-07-03","2024-07-05"]
        rows = [f"{d},{t},{value}" for d, stock in zip(dates,[10,11,12])
                for t,value in [("US",stock),("SPY",100),("TRY=X",30)]]
        prices = ("Date,Ticker,Adjusted close\n" + "\n".join(rows)).encode()
        with patch("market_data.yf.download",side_effect=AssertionError("Unexpected download")) as remote:
            app = AppTest.from_file(str(ROOT/"merhaba.py"),default_timeout=30).run()
            app.file_uploader(key="project_file").upload("snapshot.json",data,"application/json").run()
            app.button(key="open_project").click().run()
            self.assertTrue(app.button(key="run").disabled)
            field = next(f for f in app.file_uploader if f.label == "Price history CSV")
            field.upload("history.csv",prices,"text/csv").run()
            app.button(key="run").click().run()
            self.assertFalse(app.error)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value,"20.00%")
            self.assertTrue(any("32.00%" in v.value and "TRY" in v.value for v in app.caption))
            self.assertTrue(any("Information ratio" in table.value.index for table in app.dataframe))
            remote.assert_not_called()

    def test_benchmark_metrics_use_common_close_endpoints(self):
        days = pd.date_range("2024-07-02", periods=4)
        values = pd.Series([100,110,132,121],index=days)
        bench = pd.Series([100,100,110],index=days[[0,1,3]])
        active = np.array([.1,0.])
        np.testing.assert_allclose(active_returns(values,bench).dropna(),active,atol=1e-12)
        metrics = benchmark_stats(values,bench)
        self.assertAlmostEqual(metrics["Tracking error"], active.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(metrics["Information ratio"], active.mean()*252/metrics["Tracking error"])
        self.assertEqual(benchmark_stats(bench,bench)["Tracking error"],0)
        self.assertTrue(np.isnan(benchmark_stats(bench,bench)["Information ratio"]))
        self.assertTrue(benchmark_stats(values.iloc[:1],bench).isna().all())
