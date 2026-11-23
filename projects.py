"""Portable saved comparisons. Files contain the logs and shared settings."""

import json
from datetime import date
from math import isfinite

import pandas as pd

from currencies import NATIVE, SUPPORTED, currency_settings
from transactions import LOG_COLUMNS, PortfolioError, cell, symbol


SKIP = "Skip"
FORMAT = "portfolio-visualizer-comparison"
SETTING_KEYS = ("start_date", "end_date", "benchmark", "cash_rate", "starting_cash",
                "account_currency", "report_currency", "currency_overrides")


def duplicate_log(frame):
    draft = frame.reindex(columns=LOG_COLUMNS, fill_value="").reset_index(drop=True).copy()
    draft.insert(0, SKIP, False)
    return draft


def skipped_rows(frame):
    return {i for i, skip in enumerate(frame[SKIP].fillna(False), 1) if skip}


def _records(frame, scenario):
    records = frame.reindex(columns=LOG_COLUMNS, fill_value="").map(cell).to_dict("records")
    if scenario:
        for record, skip in zip(records, frame[SKIP].fillna(False)):
            record[SKIP] = bool(skip)
    return records


def save_project(original, scenario, settings):
    cfg = {key: settings[key] for key in SETTING_KEYS}
    cfg["start_date"] = cfg["start_date"].isoformat()
    cfg["end_date"] = cfg["end_date"].isoformat()
    payload = {"format": FORMAT, "version": 1, "settings": cfg, "original": _records(original, False),
               "scenario": None if scenario is None else _records(scenario, True)}
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _frame(records, scenario=False):
    if not isinstance(records, list):
        raise PortfolioError("A saved log must contain a list of rows.")
    for record in records:
        if not isinstance(record, dict) or any(not isinstance(record.get(key), str) for key in LOG_COLUMNS):
            raise PortfolioError("A saved transaction row is missing a field or contains an invalid value.")
        if scenario and not isinstance(record.get(SKIP), bool):
            raise PortfolioError("A scenario row needs a true or false Skip value.")
    frame = pd.DataFrame(records, columns=LOG_COLUMNS, dtype="string")
    if scenario:
        frame.insert(0, SKIP, pd.Series([r[SKIP] for r in records], dtype=bool))
    return frame


def read_project(data):
    try:
        saved = json.loads(data.decode("utf-8-sig"))
        if not isinstance(saved, dict) or saved.get("format") != FORMAT or saved.get("version") != 1:
            raise PortfolioError("Choose a Portfolio Visualizer comparison file (version 1).")
        settings = {key: saved["settings"][key] for key in SETTING_KEYS}
        for key in ("start_date", "end_date"):
            settings[key] = date.fromisoformat(settings[key])
        if not date(1900, 1, 1) <= settings["start_date"] <= settings["end_date"] <= date.today():
            raise PortfolioError("The saved reporting dates are outside the supported range.")
        for key in ("starting_cash", "cash_rate"):
            if type(settings[key]) not in (float, int) or not isfinite(settings[key]):
                raise PortfolioError("The saved cash settings must be finite numbers.")
        if settings["starting_cash"] <= 0 or not 0 <= settings["cash_rate"] <= 100:
            raise PortfolioError("Check the starting balance and interest rate in the saved settings.")
        if not all(isinstance(settings[k], str) for k in ("benchmark", "account_currency", "report_currency", "currency_overrides")):
            raise PortfolioError("Invalid benchmark or currency settings in the saved comparison.")
        benchmark = symbol(settings["benchmark"])
        if not benchmark or "," in benchmark or any(c.isspace() for c in benchmark):
            raise PortfolioError("The saved comparison needs one benchmark ticker.")
        if settings["account_currency"] not in (*SUPPORTED, NATIVE) or settings["report_currency"] not in (*SUPPORTED, NATIVE):
            raise PortfolioError("Unsupported saved currency.")
        currency_settings(settings["account_currency"], settings["report_currency"], settings["currency_overrides"])
        original = _frame(saved["original"])
        scenario = _frame(saved["scenario"], True) if saved["scenario"] is not None else None
    except PortfolioError:
        raise
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise PortfolioError("Could not open this comparison file. Choose a JSON file saved by this app.") from exc
    return original, scenario, settings
