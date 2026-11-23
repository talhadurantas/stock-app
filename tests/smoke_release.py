"""Exercise the deployed public app after a merge, using only example transactions."""

import re
import time
from datetime import date

from playwright.sync_api import expect, sync_playwright

from merhaba import EXAMPLES, table
from projects import duplicate_log, save_project


APP_URL = "https://stock-app-drkienpnjsbecpxcxo3ina.streamlit.app/"


def live_frame(page):
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
    deadline = time.monotonic() + 300
    refresh_at = time.monotonic() + 40
    woke = False
    while time.monotonic() < deadline:
        for frame in page.frames:
            if frame.get_by_text("Logarithmic growth chart", exact=True).count():
                return frame
            wake = frame.get_by_role("button", name="Yes, get this app back up!")
            if not woke and wake.count() and wake.is_visible():
                wake.click()
                woke = True
            if time.monotonic() >= refresh_at and frame.get_by_role("heading", name="Portfolio Visualizer").count():
                page.reload(wait_until="domcontentloaded")
                refresh_at = time.monotonic() + 40
                break
        page.wait_for_timeout(1500)
    raise AssertionError("The deployed app did not show the scenario controls within five minutes.")


def replay(frame, original, skip, start, end, initial, rate, *, name, report="USD"):
    variant = duplicate_log(original)
    variant.loc[skip - 1, "Skip"] = True
    settings = dict(start_date=start, end_date=end, benchmark="SPY", cash_rate=rate, starting_cash=initial,
                    account_currency="USD", report_currency=report, currency_overrides="")
    blob = save_project(original, variant, settings)
    open_button = frame.get_by_role("button", name="Open saved comparison", exact=True)
    if not open_button.is_visible():
        frame.get_by_text("Save or open a comparison", exact=True).click()
    uploader = frame.locator('[data-testid="stFileUploader"]').filter(has_text="Saved comparison JSON")
    uploader.locator('input[type=file]').set_input_files(
        {"name": f"{name}.json", "mimeType": "application/json", "buffer": blob})
    expect(uploader.get_by_text(f"{name}.json", exact=True)).to_be_visible(timeout=20_000)
    expect(open_button).to_be_enabled(timeout=20_000)
    open_button.click()
    expect(frame.get_by_text("Opened the saved comparison. Run analysis to replay it.", exact=True)).to_be_visible(timeout=25_000)
    comparison = frame.get_by_text(re.compile("^Scenario minus original:"))
    expect(comparison).to_have_count(0, timeout=20_000)
    run = frame.get_by_role("button", name="Run analysis", exact=True)
    for attempt in range(2):
        run.click()
        try:
            expect(comparison).to_be_visible(timeout=45_000)
            break
        except AssertionError:
            if attempt or not frame.get_by_text(re.compile("Yahoo Finance|automatic retry")).count():
                print(frame.locator('[data-testid="stAlert"]').all_text_contents(), flush=True)
                raise
    expect(frame.get_by_role("heading", name="Account value and investment gain")).to_be_visible()
    chart = frame.locator("canvas.marks, svg.marks").first
    expect(chart).to_be_visible(timeout=25_000)
    return comparison.inner_text()


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        frame = live_frame(page)
        cash = table([("2024-03-04", "Initial", "", "", ""),
                      ("2024-03-05", "Deposit", "", "", ""),
                      ("2024-03-06", "Fee", "", "", "")])
        cash.loc[1, "Amount"] = "100"
        cash.loc[2, "Amount"] = "10"
        result = replay(frame, cash, 3, date(2024,3,4), date(2024,3,6), 1000, 0, name="cash")
        assert "+0.91" in result, result
        print("Flat-market cash/fee comparison:", result, flush=True)
        tech = replay(frame, table(EXAMPLES["Tech growth"]), 2, date(2013,1,1), date(2024,12,31), 100, 4.5, name="tech")
        print("Tech growth, skip the 2017 switch, through 2024-12-31:", tech, flush=True)
        mixed = table([("2024-03-04", "Initial", "", "AAPL,THYAO.IS", "40,40"),
                       ("2024-03-06", "Fee", "", "", "")])
        mixed.loc[1, "Amount"] = "10"
        lira = replay(frame, mixed, 2, date(2024,3,4), date(2024,3,6), 1000, 0, name="istanbul", report="TRY")
        expect(frame.get_by_text(re.compile("^Amounts in TRY"))).to_be_visible()
        print("Mixed AAPL and THYAO.IS with the benchmark in TRY:", lira, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
