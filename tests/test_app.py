import unittest
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "merhaba.py"
OLD_EXPORT = b'Date,Action,Sell,Buy,Weights\n2024-01-08,Initial,,"AAA,BBB","40,40"\n'


class InterfaceTests(unittest.TestCase):
    def test_import_analyze_rerun_and_reject_a_bad_replacement(self):
        days = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"])
        sample = pd.DataFrame({"AAA": [10, 11, 11], "BBB": [20, 22, 22], "SPY": [100]*3}, index=days)
        with patch("market_data.yf.download", return_value=pd.concat({"Close": sample}, axis=1)):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.file_uploader(key="csv_file").upload("old.csv", OLD_EXPORT, "text/csv").run()
            app.button(key="import_log").click().run()
            self.assertEqual(app.data_editor[0].value.iloc[0]["Buy"], "AAA,BBB")
            app.date_input(key="start_date").set_value(date(2024, 1, 8))
            app.date_input(key="end_date").set_value(date(2024, 1, 10))
            app.number_input(key="cash_rate").set_value(0.0)
            app.button(key="run").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[0].value, "8.00%")
            self.assertEqual({b.label for b in app.download_button}, {
                "Download transaction CSV", "Daily ledger CSV", "Transaction CSV", "Summary report"})
            app.checkbox[0].uncheck().run()
            self.assertEqual(app.metric[0].value, "8.00%")
            app.file_uploader(key="csv_file").upload("bad.csv", b"Date,Amount\n2024-01-01,10", "text/csv").run()
            app.button(key="import_log").click().run()
            self.assertTrue(app.error)
            self.assertEqual(app.data_editor[0].value.iloc[0]["Buy"], "AAA,BBB")
            self.assertEqual(app.metric[0].value, "8.00%")

    def test_example_replacement_and_clear_do_not_leave_edit_rows_behind(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.data_editor[0].value), 4)
        next(b for b in app.button if b.label == "Value play example").click().run()
        self.assertEqual(len(app.data_editor[0].value), 3)
        next(b for b in app.button if b.label == "Add row").click().run()
        self.assertEqual(len(app.data_editor[0].value), 4)
        self.assertTrue(app.button(key="run").disabled)
        next(b for b in app.button if b.label == "Clear log").click().run()
        self.assertEqual(len(app.data_editor[0].value), 0)
        self.assertTrue(app.button(key="run").disabled)
        self.assertFalse(app.exception)

    def test_partial_sale_comparison_warning_and_downloads(self):
        source = (b"Date,Action,Sell,Buy,Weights,Sell %\n"
                  b"2024-02-05,Initial,,AAA,100,\n"
                  b"2024-02-05,Buy,,SKIP,50,\n"
                  b"2024-02-06,Sell & Buy,AAA,BBB,60,25\n")
        days = pd.date_range("2024-02-05", periods=3)
        prices = pd.DataFrame({"AAA": [100, 200, 100], "BBB": [100]*3, "SPY": [100, 99, 98]}, index=days)
        media = MemoryMediaFileStorage("/mock/media")
        with (patch("market_data.yf.download", return_value=pd.concat({"Close": prices}, axis=1)),
              patch("streamlit.testing.v1.app_test.MemoryMediaFileStorage", return_value=media)):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.file_uploader(key="csv_file").upload("partial.csv", source, "text/csv").run()
            app.button(key="import_log").click().run()
            self.assertEqual(app.data_editor[0].value["Sell %"].iloc[-1], "25")
            app.date_input(key="start_date").set_value(days[0].date())
            app.date_input(key="end_date").set_value(days[-1].date())
            app.number_input(key="cash_rate").set_value(0.0)
            app.button(key="run").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            comparison = next(x.value for x in app.dataframe if "Strategy" in x.value)
            self.assertEqual(comparison["Strategy"].tolist(), ["Portfolio", "Buy and hold", "SPY (benchmark)"])
            self.assertEqual(comparison["Ending value per 100"].tolist(), ["125.00", "100.00", "98.00"])
            self.assertEqual(comparison["Max drawdown"].tolist(), ["-37.50%", "-50.00%", "-2.00%"])
            self.assertTrue(any("Row 2" in w.value and "SKIP" in w.value and "No cash available" in w.value
                                for w in app.warning))

            def downloaded(key):
                filename = Path(app.download_button(key=key).proto.url).name
                return media.get_file(filename).content

            self.assertEqual(downloaded("result_log"), source)
            ledger = pd.read_csv(BytesIO(downloaded("ledger_csv")))
            self.assertEqual(ledger["Buy and Hold Value"].tolist(), [100, 200, 100])
            self.assertEqual(ledger["Buy and Hold Daily Return"].iloc[-1], -0.5)
            self.assertEqual(ledger["Cash Value"].tolist(), [0, 20, 20])
            report = downloaded("report_txt").decode()
            for value in ("Buy and hold", "Ending value per 100", "No cash available", "AAA 25%"):
                self.assertIn(value, report)


if __name__ == "__main__":
    unittest.main()
