"""Run the real Streamlit UI with deterministic quotes in browser tests."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from merhaba import main


def sample_prices(tickers, *, start, end, **kwargs):
    dates = pd.bdate_range(start, end, inclusive="left")
    steps = np.arange(len(dates), dtype=float)
    quotes = {ticker: 100 + steps * (i + 1) / 100 for i, ticker in enumerate(tickers)}
    return pd.concat({"Close": pd.DataFrame(quotes, index=dates)}, axis=1)


with patch("market_data.yf.download", side_effect=sample_prices):
    main()
