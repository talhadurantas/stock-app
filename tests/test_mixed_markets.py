import unittest

import numpy as np
import pandas as pd

from currencies import Currencies, FX_TICKER
from portfolio import COLUMNS, PortfolioError, backtest, parse_log
from market_data import analyze_scenarios
from transactions import Trade


DAYS = pd.bdate_range("2024-07-02", "2024-07-16")
US_HOLIDAY = pd.Timestamp("2024-07-04")
TR_HOLIDAY = pd.Timestamp("2024-07-15")


def markets():
    prices = pd.DataFrame({"US": 10., "TR.IS": 30., "SPY": 100., "XU100.IS": 300., FX_TICKER: 3.}, index=DAYS)
    prices.loc[US_HOLIDAY, ["SPY", "US"]] = np.nan
    prices.loc[TR_HOLIDAY, ["TR.IS", "XU100.IS"]] = np.nan
    return prices


def replay(rows, *, prices=None, bench="SPY", end=None, rate=0):
    logs = parse_log(pd.DataFrame(rows, columns=COLUMNS))
    return backtest(markets() if prices is None else prices, logs, bench, DAYS[0], end or DAYS[-1], rate)


class MarketTests(unittest.TestCase):
    def test_us_holiday_price_jump_is_independent_of_benchmark(self):
        prices = markets()
        prices.loc["2024-07-05":, "TR.IS"] *= 1.20
        rows = [("2024-07-02","Initial","","",""), ("2024-07-04","Buy","","TR.IS","100")]
        a = replay(rows, prices=prices)
        b = replay(rows, prices=prices, bench="XU100.IS")
        for result in (a, b):
            self.assertEqual(result.executions.Executed.iloc[-1], US_HOLIDAY)
            self.assertAlmostEqual(result.ledger.Portfolio.iloc[-1], 120)
        pd.testing.assert_series_equal(a.ledger.Portfolio, b.ledger.Portfolio)
        self.assertNotIn(US_HOLIDAY, a.benchmark_closes)
        self.assertEqual(a.ledger.Benchmark.loc[US_HOLIDAY], 100)

    def test_istanbul_holiday_rolls_only_its_order(self):
        rows = [("2024-07-02","Initial","","",""), ("2024-07-15","Buy","","TR.IS","50"),
                ("2024-07-15","Buy","","US","50")]
        r = replay(rows)
        events = r.executions.set_index("Log row")
        self.assertEqual(events.at[2,"Executed"],pd.Timestamp("2024-07-16"))
        self.assertEqual(events.at[3,"Executed"],TR_HOLIDAY)
        self.assertAlmostEqual(events.at[2,"Account bought"],25)
        gap = markets()
        gap.loc["2024-07-16", "TR.IS"] = np.nan
        with self.assertRaisesRegex(PortfolioError, "Row 2.*no fresh adjusted close for TR.IS on 2024-07-16"):
            replay(rows, prices=gap)

    def test_late_sale_reserves_proceeds_for_next_istanbul_close(self):
        rows = [("2024-07-02","Initial","","US","100"),
                ("2024-07-05","Sell & Buy","US","TR.IS","100"),
                ("2024-07-05","Buy","","US","100")]
        r = replay(rows)
        switch = r.executions.loc[r.executions["Log row"] == 2]
        self.assertEqual(switch.Leg.tolist(), ["Sell","Buy"])
        self.assertEqual(switch.Executed.dt.strftime("%Y-%m-%d").tolist(), ["2024-07-05","2024-07-08"])
        self.assertEqual(switch["Close (UTC)"].dt.hour.tolist(), [20,15])
        self.assertEqual(r.no_orders.Note.tolist(), ["No cash available"])
        self.assertAlmostEqual(r.account.Cash.loc["2024-07-05"],100)
        self.assertAlmostEqual(r.positions["TR.IS"].iloc[-1],100)
        partial = replay(rows, end=pd.Timestamp("2024-07-05"))
        self.assertEqual(partial.pending,(2,))
        self.assertEqual(partial.account.Cash.iloc[-1],100)

    def test_early_close_sale_can_fund_later_same_day_purchase(self):
        r = replay([("2024-07-02","Initial","","TR.IS","100"), ("2024-07-03","Sell & Buy","TR.IS","US","100")])
        switch = r.executions.loc[r.executions["Log row"] == 2]
        self.assertEqual(switch["Close (UTC)"].dt.hour.tolist(), [15,17])
        self.assertEqual(switch.Executed.nunique(),1)
        self.assertAlmostEqual(r.positions.US.iloc[-1],100)

    def test_basket_on_us_holiday_and_cross_market_rebalance(self):
        rows = [("2024-07-04","Initial","","TR.IS,US","30,50"), ("2024-07-09","Rebalance","","TR.IS,US","25,75")]
        r = replay(rows)
        first = r.executions[r.executions["Log row"] == 1]
        self.assertEqual(first["Account bought"].tolist(), [30,50])
        self.assertAlmostEqual(r.account.Cash.loc["2024-07-04"],70)
        np.testing.assert_allclose(r.positions.iloc[-1][["TR.IS","US"]], [25,75])
        np.testing.assert_allclose(r.ledger.Portfolio,100)
        np.testing.assert_allclose(r.ledger["Buy and hold"],100)
        self.assertAlmostEqual(r.contributions.sum().sum(),0)

    def test_overlapping_rebalances_wait_for_the_preceding_basket(self):
        rows = [("2024-07-02","Initial","","TR.IS,US","50,50"),
                ("2024-07-05","Rebalance","","TR.IS,US","30,70"),
                ("2024-07-05","Rebalance","","TR.IS,US","60,40")]
        r = replay(rows)
        old = r.executions.loc[r.executions["Log row"] == 2, "Close (UTC)"]
        new = r.executions.loc[r.executions["Log row"] == 3, "Close (UTC)"]
        self.assertGreaterEqual(new.min(), old.max())
        self.assertAlmostEqual(r.positions["TR.IS"].iloc[-1],60)
        self.assertAlmostEqual(r.positions["US"].iloc[-1],40)
        np.testing.assert_allclose(r.ledger.Portfolio,100)

    def test_cash_clock_and_shared_valuation_dates_survive_benchmark_and_scenario_changes(self):
        original = [Trade(1,DAYS[0],"Initial",(),(),())]
        variant = [Trade(2,US_HOLIDAY,"Buy",(),("TR.IS",),(0.5,))]
        def run(bench):
            return analyze_scenarios(original, variant, bench, DAYS[0], DAYS[-1], 10, 100, Currencies(),
                                     cached_download=lambda *_: markets())
        a, scenario, error = run("SPY")
        b, _, _ = run("XU100.IS")
        self.assertIsNone(error)
        self.assertEqual(a.ledger.index.tolist(),scenario.ledger.index.tolist())
        pd.testing.assert_series_equal(a.ledger.Portfolio,b.ledger.Portfolio)
        self.assertEqual(a.account.Cash.loc[US_HOLIDAY], a.account.Cash.loc["2024-07-03"])

    def test_uploaded_history_never_retries_yahoo_and_preserves_a_good_original(self):
        from unittest.mock import patch
        from market_data import read_price_history

        data = b"Date,Ticker,Adjusted close\n2024-07-02,US,100\n2024-07-03,US,120\n2024-07-02,SPY,10\n2024-07-03,SPY,10\n"
        prices = read_price_history(data)
        original = [Trade(1,DAYS[0],"Initial",(),("US",),(1.,))]
        missing = [Trade(1,DAYS[0],"Initial",(),("ABSENT",),(1.,))]
        with patch("market_data.yf.download",side_effect=AssertionError("Unexpected network request")) as fetch:
            r, alt, issue = analyze_scenarios(original,missing,"SPY",DAYS[0],DAYS[1],0,100,Currencies(),price_history=prices)
            self.assertEqual(r.ledger.Portfolio.iloc[-1],120)
            self.assertIsNone(alt)
            self.assertIn("uploaded price history CSV",issue)
            with self.assertRaisesRegex(PortfolioError, "uploaded history ends on 2024-07-03"):
                analyze_scenarios(original,None,"SPY",DAYS[0],DAYS[-1],0,100,Currencies(),price_history=prices)
            fetch.assert_not_called()
        with self.assertRaisesRegex(PortfolioError,"one adjusted close"):
            read_price_history(data+b"2024-07-03,US,125\n")
