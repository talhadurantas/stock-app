import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from market_data import analyze_portfolio
from portfolio import COLUMNS, QuoteError, parse_log


DAYS = pd.date_range("2024-01-08", periods=3)
FIRST = ("2024-01-08", "Initial", "", "A", "100")


def response(values, days=DAYS):
    return pd.concat({"Close": pd.DataFrame(values, index=days)}, axis=1)


def run_log(rows=(FIRST,), **kwargs):
    trades = parse_log(pd.DataFrame(rows, columns=COLUMNS))
    return analyze_portfolio(trades, "SPY", DAYS[0], DAYS[-1], 0, **kwargs)


@patch("market_data.yf.download")
class RetryTests(unittest.TestCase):
    def test_success_needs_one_request_and_a_zero_budget_order_needs_no_price(self, download):
        download.return_value = response({"A": [10, 11, 12], "SPY": [100]*3})
        result = run_log([FIRST, ("2024-01-09", "Buy", "", "UNQUOTED", "100")])
        self.assertEqual(download.call_count, 1)
        self.assertEqual(result.no_orders.Note.tolist(), ["No cash available"])
        np.testing.assert_allclose(result.ledger.Portfolio, [100, 110, 120])

    def test_missing_benchmark_refetches_only_it_and_preserves_original_quotes(self, download):
        cached = Mock(return_value=pd.DataFrame({"A": [10, 12]}, index=DAYS[[0, 2]]))
        clear = Mock()
        download.return_value = response({"SPY": [100, 99, 101]})
        result = run_log(cached_download=cached, clear_cache=clear)
        self.assertEqual([c.args[0] for c in download.call_args_list], [["SPY"]])
        clear.assert_called_once_with(*cached.call_args.args)
        np.testing.assert_allclose(result.ledger.Portfolio, [100, 100, 120])
        np.testing.assert_allclose(result.ledger.Benchmark, [100, 99, 101])
        self.assertEqual(result.carried, {"A": 1})

    def test_required_trade_close_is_repaired_without_erasing_existing_closes(self, download):
        download.side_effect = [
            response({"A": [10, np.nan, 15], "B": [10, 10, np.nan], "SPY": [100]*3}),
            response({"A": [20]}, DAYS[1:2]),
            response({"B": [11]}, DAYS[2:]),
        ]
        result = run_log([FIRST, ("2024-01-09", "Sell & Buy", "A", "B", "100"),
                          ("2024-01-10", "Sell", "B", "", "")])
        self.assertEqual([c.args[0] for c in download.call_args_list], [["A", "B", "SPY"], ["A"], ["B"]])
        np.testing.assert_allclose(result.ledger.Portfolio, [100, 200, 220])
        np.testing.assert_allclose(result.ledger["Buy and hold"], [100, 200, 150])

    def test_transient_whole_request_failure_retries_once(self, download):
        download.side_effect = [OSError("connection lost"), response({"A": [10]*3, "SPY": [100]*3})]
        np.testing.assert_allclose(run_log().ledger.Portfolio, 100)
        self.assertEqual([c.args[0] for c in download.call_args_list], [["A", "SPY"]]*2)

    def test_persistent_gap_reports_row_date_and_ticker_without_a_third_request(self, download):
        download.side_effect = [response({"A": [np.nan, 10, 10], "SPY": [100]*3}),
                                response({"A": [np.nan, 10, 10]})]
        with self.assertRaisesRegex(QuoteError, "Row 1.*A on 2024-01-08.*after one automatic retry"):
            run_log()
        self.assertEqual([c.args[0] for c in download.call_args_list], [["A", "SPY"], ["A"]])

    def test_empty_targeted_retry_retains_the_original_trade_context(self, download):
        download.side_effect = [response({"SPY": [100]*3}), pd.DataFrame()]
        with self.assertRaisesRegex(QuoteError, "Row 1.*A on 2024-01-08.*after one automatic retry"):
            run_log()
        self.assertEqual(download.call_count, 2)

    def test_empty_provider_response_and_invalid_prices_have_bounded_requests(self, download):
        download.return_value = pd.DataFrame()
        with self.assertRaisesRegex(QuoteError, "no prices.*after one automatic retry"):
            run_log()
        self.assertEqual(download.call_count, 2)
        download.reset_mock()
        download.return_value = response({"A": [0, 1, 2], "SPY": [100]*3})
        with self.assertRaisesRegex(QuoteError, "positive and finite"):
            run_log()
        self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
