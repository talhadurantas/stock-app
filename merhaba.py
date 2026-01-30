import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="Pro Portfolio Visualizer", layout="wide")
st.title("📈 Pro Stock Portfolio Visualizer")

# --- Caching Function ---
@st.cache_data(ttl=3600)
def get_stock_data(tickers, benchmark, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    bench_data = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=False)
    return data, bench_data

# --- Sidebar ---
with st.sidebar:
    st.header("Portfolio Settings")
    
    # 1. Ticker Input (Formun dışında bırakıyoruz ki enter'a basınca slider'lar oluşsun)
    ticker_string = st.text_input("Enter Stock Tickers (comma separated)", value="AAPL, MSFT, GOOGL")
    tickers = [x.strip().upper() for x in ticker_string.split(',') if x.strip()]
    
    # --- FORM BAŞLANGICI (Mobil hatayı çözen kısım burası) ---
    with st.form(key='my_form'):
        st.subheader("Asset Allocation")
        weights = []
        if tickers:
            st.info("Adjust weights below.")
            for t in tickers:
                val = st.slider(f"Weight for {t}", 0, 100, 50, key=t)
                weights.append(val)
        
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
        else:
            weights = [1.0 / len(tickers)] * len(tickers)

        st.subheader("Benchmark & Timeframe")
        benchmark = st.text_input("Benchmark Ticker", value="SPY").upper()
        
        # Tarih seçiciler artık formun içinde olduğu için mobilde hata vermez
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))
        
        # Buton artık bir "Form Submit" butonu
        run_btn = st.form_submit_button("🚀 Run Analysis")

# --- Analysis Logic ---
if run_btn:
    if not tickers:
        st.error("Please enter at least one stock ticker.")
    else:
        try:
            with st.spinner("Fetching data..."):
                raw_data, bench_raw = get_stock_data(tickers, benchmark, start_date, end_date)

            if raw_data.empty or bench_raw.empty:
                st.error("No data found. Try again.")
                st.stop()

            # --- Data Processing ---
            if 'Adj Close' in raw_data.columns:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                data = raw_data['Close']
            else:
                data = raw_data
            
            if 'Adj Close' in bench_raw.columns:
                bench_data = bench_raw['Adj Close']
            elif 'Close' in bench_raw.columns:
                bench_data = bench_raw['Close']
            else:
                bench_data = bench_raw

            if len(tickers) == 1:
                if isinstance(data, pd.Series):
                    data = data.to_frame(tickers[0])
            
            daily_returns = data.pct_change().dropna()
            bench_returns = bench_data.pct_change().dropna()
            
            common_index = daily_returns.index.intersection(bench_returns.index)
            daily_returns = daily_returns.loc[common_index]
            bench_returns = bench_returns.loc[common_index]
            
            if daily_returns.empty:
                st.error("Timestamps didn't match.")
                st.stop()

            if len(tickers) > 1:
                portfolio_daily = (daily_returns * weights).sum(axis=1)
            else:
                portfolio_daily = daily_returns.iloc[:, 0]
                
            portfolio_cum = (1 + portfolio_daily).cumprod()
            
            if isinstance(bench_returns, pd.DataFrame):
                bench_returns = bench_returns.iloc[:, 0]
            bench_cum = (1 + bench_returns).cumprod()
            
            total_return = portfolio_cum.iloc[-1] - 1
            bench_total_return = bench_cum.iloc[-1] - 1
            
            col1, col2 = st.columns(2)
            col1.metric("Your Return", f"{total_return:.2%}")
            col2.metric("Benchmark Return", f"{bench_total_return:.2%}")

            st.subheader("Growth Chart ($1 Investment)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(portfolio_cum.index, portfolio_cum, label='Your Portfolio', linewidth=2)
            ax.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark})', linestyle='--', color='gray')
            ax.axhline(1.0, color='red', linestyle=':', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
