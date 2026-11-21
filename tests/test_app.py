import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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


if __name__ == "__main__":
    unittest.main()
