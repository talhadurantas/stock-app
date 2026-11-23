import json
import unittest
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

from analytics import drawdown, holding_breakdown, stats
from charts import growth_chart, trade_markers
from currencies import Currencies, FX_TICKER, currency_settings
from market_data import analyze_scenarios
from portfolio import BUY_AND_HOLD, LOG_COLUMNS, PortfolioError, backtest, export_log, import_log, parse_log
from projects import duplicate_log, read_project, save_project, skipped_rows


DAYS = pd.date_range("2024-03-04", periods=4)
APP = Path(__file__).resolve().parents[1] / "merhaba.py"


def frame(rows):
    return pd.DataFrame([dict(zip(LOG_COLUMNS, row)) for row in rows], columns=LOG_COLUMNS).fillna("").astype("string")


def run(rows, data, *, start=None, **settings):
    quotes = pd.DataFrame(data, index=DAYS[:len(next(iter(data.values())))])
    if "SPY" not in quotes:
        quotes["SPY"] = 100.0
    return backtest(quotes, parse_log(frame(rows)), "SPY", start or quotes.index[0], quotes.index[-1], 0, **settings)


def project_settings():
    return dict(start_date=DAYS[0].date(), end_date=DAYS[2].date(), benchmark="SPY", cash_rate=0.0,
                starting_cash=1000.0, account_currency="USD", report_currency="USD", currency_overrides="")


class AccountingTests(unittest.TestCase):
    def test_flat_market_deposit_withdrawal_and_fee(self):
        rows = [
            ("2024-03-04", "Initial", "", "", ""),
            ("2024-03-05", "Deposit", "", "", "", "", "10000"),
            ("2024-03-06", "Withdrawal", "", "", "", "", "10000"),
            ("2024-03-07", "Fee", "", "", "", "", "50"),
        ]
        result = run(rows, {"SPY": [100]*4}, initial_cash=50000, currencies=Currencies("TRY", "TRY", (("SPY","TRY"),)))
        np.testing.assert_allclose(result.account.Balance, [50000, 60000, 50000, 49950])
        np.testing.assert_allclose(result.ledger.Portfolio, [100, 100, 100, 99.9])
        np.testing.assert_allclose(result.account["Investment gain"], [0, 0, 0, -50])
        self.assertAlmostEqual(result.contributions.Fees.sum(), -0.1)
        self.assertTrue(result.no_orders.empty)
        self.assertIn(b"Amount", export_log(frame(rows)).splitlines()[0])
        self.assertEqual(parse_log(import_log(export_log(frame(rows))))[-1].amount, 50)

    def test_zero_opening_balance_can_be_funded_with_the_first_deposit(self):
        result = run([("2024-03-04","Deposit","","","","","1000"),
                      ("2024-03-04","Buy","","A","100")], {"A":[10,11]}, initial_cash=0)
        np.testing.assert_allclose(result.ledger.Portfolio, [100,110])
        np.testing.assert_allclose(result.account["Net contributed"], 1000)
        self.assertAlmostEqual(result.account["Investment gain"].iloc[-1],100)

    def test_first_close_fee_is_in_return_drawdown_and_attribution(self):
        result = run([("2024-03-04","Initial","","A","100","","","10")], {"A":[100,100]}, initial_cash=1000)
        np.testing.assert_allclose(result.account.Balance, 990)
        np.testing.assert_allclose(result.ledger[BUY_AND_HOLD], 99)
        self.assertAlmostEqual(stats(result.ledger.Portfolio, opening=100)["Total return"], -0.01)
        detail = holding_breakdown(result)
        np.testing.assert_allclose(detail.loc["Fees"], [-1,-1])
        self.assertAlmostEqual(result.executions["Account bought"].iloc[0], 990)

    def test_sale_fee_and_timed_flows_reconcile_in_a_sliced_window(self):
        rows = [
            ("2024-03-04","Initial","","A","80"),
            ("2024-03-05","Deposit","","","","","500","10"),
            ("2024-03-05","Sell & Buy","A","B","100","25","","5"),
            ("2024-03-06","Withdrawal","","","","","100"),
        ]
        data = {"A":[100,200,100,110], "B":[100,100,90,95]}
        result = run(rows, data, initial_cash=1000)
        self.assertAlmostEqual(result.executions["Account sold"].iloc[2],400)
        self.assertAlmostEqual(result.executions["Account bought"].iloc[2],395)
        for view in (result, run(rows, data, initial_cash=1000, start=DAYS[1])):
            detail = holding_breakdown(view)
            self.assertAlmostEqual(detail.iloc[:,0].sum(), view.ledger.Portfolio.iloc[-1]-100)
            self.assertAlmostEqual(detail.iloc[:,1].sum(), 100*drawdown(view.ledger.Portfolio,100)[0].min())
            np.testing.assert_allclose(view.account.Balance-view.account["Net contributed"], view.account["Investment gain"])

    def test_full_withdrawal_and_refunding_hold_the_index_while_empty(self):
        rows = [("2024-03-04","Initial","","",""), ("2024-03-05","Withdrawal","","","","","100"),
                ("2024-03-07","Deposit","","","","","50")]
        result = run(rows, {"SPY":[100]*4,FX_TICKER:[10,20,30,40]}, currencies=Currencies("USD","TRY"))
        np.testing.assert_allclose(result.account.Balance, [1000,0,0,2000])
        np.testing.assert_allclose(result.ledger.Portfolio, [100,200,200,200])
        self.assertAlmostEqual(result.contributions.Cash.sum(), 100)

    def test_insufficient_cash_and_invalid_amounts_name_the_row(self):
        with self.assertRaisesRegex(PortfolioError, "Row 2.*withdrawal"):
            run([("2024-03-04","Initial","","A","100"),("2024-03-05","Withdrawal","","","","","10")],{"A":[10,11]})
        for row in [("2024-03-04","Deposit","","","","","nan"), ("2024-03-04","Buy","","A","","","100"),
                    ("2024-03-04","Fee","","","","","100","10"), ("2024-03-04","Buy","","A","","","","-1")]:
            with self.subTest(row=row), self.assertRaisesRegex(PortfolioError,"Row 1"):
                parse_log(frame([row]))

    def test_mixed_us_istanbul_portfolio_and_benchmark_in_both_currencies(self):
        rows = [("2024-03-04","Initial","","A,TR.IS","50,50")]
        data = {"A":[100]*3, "TR.IS":[1000,np.nan,1200], "SPY":[100]*3, FX_TICKER:[10,20,20]}
        usd = run(rows, data)
        lira = run(rows, data, currencies=Currencies("USD","TRY"))
        np.testing.assert_allclose(usd.ledger.Portfolio,[100,75,80])
        np.testing.assert_allclose(lira.ledger.Portfolio,[100,150,160])
        np.testing.assert_allclose(lira.ledger.Benchmark,[100,200,200])
        np.testing.assert_allclose(lira.account.Balance,usd.account.Balance * np.array([10,20,20]))
        np.testing.assert_allclose(lira.contributions[["A","TR.IS"]].sum(),[50,10])
        self.assertEqual(usd.carried, {"TR.IS":1})

    def test_fx_uses_only_rates_known_at_the_close_and_respects_overrides(self):
        rows = [("2024-03-04","Initial","","XYZ","")]
        config = currency_settings("USD","USD","XYZ=TRY")
        with self.assertRaisesRegex(PortfolioError, "(?i)no USD/TRY rate on or before 2024-03-04"):
            run(rows, {"XYZ":[100,110],FX_TICKER:[np.nan,10]}, currencies=config)
        r = run(rows, {"XYZ":[100,110,110],FX_TICKER:[10,np.nan,11]}, currencies=config)
        np.testing.assert_allclose(r.ledger.Portfolio,[100,110,100])
        # No investment in lira until the final day. An earlier unavailable FX rate is harmless.
        later = [("2024-03-04","Initial","","",""),("2024-03-06","Buy","","TR.IS","100")]
        delayed = run(later, {"TR.IS":[10,11,12],FX_TICKER:[np.nan,np.nan,10]})
        np.testing.assert_allclose(delayed.ledger.Portfolio,100)


class ScenarioTests(unittest.TestCase):
    def test_shared_data_edited_copy_and_row_specific_dependency_error(self):
        original = frame([("2024-03-04","Initial","","A",""),("2024-03-05","Sell & Buy","A","B",""),
                          ("2024-03-06","Sell","B","","")])
        variant = duplicate_log(original)
        variant.loc[1,"Skip"] = True
        quotes = pd.DataFrame({"A":[100,110,120],"B":[100,100,105],"SPY":[100]*3},index=DAYS[:3])
        fetch = Mock(return_value=quotes)
        def replay():
            return analyze_scenarios(parse_log(original),parse_log(variant,skip_rows=skipped_rows(variant),allow_empty=True),
                                     "SPY",DAYS[0],DAYS[2],0,100,Currencies(),cached_download=fetch)
        base, alt, issue = replay()
        self.assertIsNone(alt)
        self.assertIn("Row 3: cannot sell an unheld position (B)",issue)
        fetch.assert_called_once()
        variant.loc[2,"Skip"] = True
        base, alt, issue = replay()
        self.assertIsNone(issue)
        self.assertAlmostEqual(alt.ledger.Portfolio.iloc[-1],120)
        self.assertAlmostEqual(base.ledger.Portfolio.iloc[-1],115.5)
        self.assertEqual(base.ledger.index.tolist(),alt.ledger.index.tolist())
        variant["Skip"] = True
        base, alt, issue = replay()
        self.assertIsNone(issue)
        np.testing.assert_allclose(alt.ledger.Portfolio,100)
        self.assertTrue(alt.executions.empty)
        self.assertEqual(original.iloc[1]["Sell"],"A")

    def test_comparison_project_round_trips_edits_skips_and_shared_settings(self):
        original = frame([("2024-03-04","Initial","","AAA","100")])
        variant = duplicate_log(original)
        variant.loc[0,["Skip","Weights"]] = [True,"50"]
        encoded = save_project(original,variant,project_settings())
        base, alt, cfg = read_project(encoded)
        pd.testing.assert_frame_equal(base,original)
        pd.testing.assert_frame_equal(alt,variant)
        self.assertEqual(cfg,project_settings())
        for content in (b"{}", b"not json", encoded.replace(b'"Skip": true',b'"Skip": "no"')):
            with self.assertRaises(PortfolioError):
                read_project(content)

    def test_chart_scale_and_hover_keep_same_day_trades(self):
        rows = [("2024-03-04","Initial","","A","50"),("2024-03-04","Buy","","A","100")]
        run_a = run(rows,{"A":[100,95]})
        spec = growth_chart(run_a,run_a,logarithmic=True,shade=True).to_dict()
        self.assertIn('"type": "log"', json.dumps(spec))
        self.assertIn("Transactions",json.dumps(spec))
        points = trade_markers(run_a,"Portfolio")
        self.assertEqual(len(points),1)
        for part in ("Row 1", "Row 2", "buy A", "cash 0.00"):
            self.assertIn(part,points.Transactions.iloc[0])


class SavedComparisonInterfaceTests(unittest.TestCase):
    def test_open_change_chart_run_and_download_a_saved_comparison(self):
        original = frame([("2024-03-04","Initial","","AAA","100"),
                          ("2024-03-05","Sell & Buy","AAA","BBB","100")])
        alt = duplicate_log(original)
        alt.loc[1,"Skip"] = True
        saved = save_project(original,alt,project_settings())
        close = pd.DataFrame({"AAA":[100,100,120],"BBB":[100,100,90],"SPY":[100]*3},index=DAYS[:3])
        media = MemoryMediaFileStorage("/mock/media")
        with (patch("market_data.yf.download",return_value=pd.concat({"Close":close},axis=1)) as fetch,
              patch("streamlit.testing.v1.app_test.MemoryMediaFileStorage",return_value=media)):
            app = AppTest.from_file(str(APP),default_timeout=40).run()
            app.file_uploader(key="project_file").upload("my_comparison.json",saved,"application/json").run()
            app.button(key="open_project").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.toggle(key="compare_scenario").value)
            self.assertEqual(app.number_input(key="starting_cash").value,1000)
            self.assertEqual(len(app.data_editor),2)
            app.button(key="run").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            comparison = next(x.value for x in app.dataframe if "Ending value per 100" in x.value)
            rows = comparison.set_index("Strategy")
            self.assertEqual(rows.at["Scenario","Ending value per 100"],"120.00")
            self.assertEqual(rows.at["Portfolio","Ending value per 100"],"90.00")
            app.checkbox(key="log_chart").check().run()
            self.assertFalse(app.exception)
            fetch.assert_called_once()
            def downloaded(key):
                return media.get_file(Path(app.download_button(key=key).proto.url).name).content
            _, alternate, cfg = read_project(downloaded("save_project"))
            self.assertEqual(skipped_rows(alternate),{2})
            self.assertEqual(cfg["starting_cash"],1000)
            ledger = pd.read_csv(BytesIO(downloaded("ledger_csv")))
            self.assertEqual(ledger["Scenario Value"].iloc[-1],120)
            self.assertEqual(ledger["Account: Balance"].iloc[-1],900)
            self.assertEqual(ledger["Return points: BBB"].iloc[-1],-10)
            app.button(key="duplicate").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.data_editor[1].value["Skip"].any())


if __name__ == "__main__":
    unittest.main()
