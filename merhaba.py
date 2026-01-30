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

# --- Date Formatting Helper ---
def format_date_input(date_str):
    """
    Automatically formats date input with real-time dash insertion:
    - User types: 2013 -> 2013
    - User types: 20130 -> 2013-0
    - User types: 201301 -> 2013-01
    - User types: 2013010 -> 2013-01-0
    - User types: 20130101 -> 2013-01-01
    Also accepts spaces, slashes, and already formatted dates.
    """
    # Remove all spaces, dashes, and slashes first
    clean = date_str.strip().replace("-", "").replace(" ", "").replace("/", "")
    
    # If it's only digits, format progressively
    if clean.isdigit():
        if len(clean) <= 4:
            # Just year: 2013
            return clean
        elif len(clean) <= 6:
            # Year + month: 2013-01
            return f"{clean[0:4]}-{clean[4:]}"
        elif len(clean) >= 7:
            # Full date: 2013-01-01
            return f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
    
    # If already formatted correctly, return as is
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str
    
    # Otherwise return original (will be validated later)
    return date_str

# --- Mobile Detection (Better UX) ---
# Automatically detect if user might be on mobile based on screen width
# This is stored in session state
if 'date_input_mode' not in st.session_state:
    st.session_state.date_input_mode = 'auto'

# --- Sidebar ---
with st.sidebar:
    st.header("Portfolio Settings")
    
    ticker_string = st.text_input("Enter Stock Tickers (comma separated)", value="AAPL, MSFT, GOOGL")
    tickers = [x.strip().upper() for x in ticker_string.split(',') if x.strip()]
    
    # --- FORM BAŞLANGICI ---
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
            weights = [1.0 / len(tickers)] * len(tickers) if tickers else [1.0]
        
        st.subheader("Benchmark & Timeframe")
        benchmark = st.text_input("Benchmark Ticker", value="SPY").upper()
        
        # --- IMPROVED DATE INPUT ---
        st.markdown("**Date Selection**")
        
        # Radio button for date input preference
        date_mode = st.radio(
            "Choose date input method:",
            ["Text Input (Mobile Friendly)", "Calendar Picker"],
            index=0,
            help="Use Text Input if calendar doesn't work on your device"
        )
        
        if date_mode == "Text Input (Mobile Friendly)":
            # Default dates
            default_start = "2013-01-01"
            default_end = str(pd.to_datetime("today").date())
            
            col1, col2 = st.columns(2)
            with col1:
                start_input = st.text_input(
                    "Start Date", 
                    value=default_start,
                    placeholder="Type: 20130101",
                    help="Just type 8 digits - dashes added automatically! (e.g., 20130101)"
                )
                # Show live preview
                start_formatted = format_date_input(start_input)
                if start_formatted != start_input and len(start_input) > 0:
                    st.caption(f"→ {start_formatted}")
                    
            with col2:
                end_input = st.text_input(
                    "End Date", 
                    value=default_end,
                    placeholder="Type: 20250130",
                    help="Just type 8 digits - dashes added automatically! (e.g., 20250130)"
                )
                # Show live preview
                end_formatted = format_date_input(end_input)
                if end_formatted != end_input and len(end_input) > 0:
                    st.caption(f"→ {end_formatted}")
            
            # Use formatted dates for validation
            try:
                start_date = pd.to_datetime(start_formatted)
                end_date = pd.to_datetime(end_formatted)
                
                # Date validation
                if start_date >= end_date:
                    st.warning("⚠️ Start date must be before end date!")
                elif end_date > pd.to_datetime("today"):
                    st.warning("⚠️ End date cannot be in the future!")
                else:
                    st.success(f"✅ Date range: {start_date.date()} to {end_date.date()}")
                    
            except Exception as e:
                st.error(f"❌ Invalid date format! Use 8 digits (20130101) or YYYY-MM-DD (2013-01-01)")
                # Fallback to default dates
                start_date = pd.to_datetime(default_start)
                end_date = pd.to_datetime(default_end)
        else:
            # Calendar picker (works better on desktop)
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date", 
                    value=pd.to_datetime("2013-01-01"),
                    help="Select start date from calendar"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=pd.to_datetime("today"),
                    help="Select end date from calendar"
                )
        
        run_btn = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

# --- Analysis Logic ---
if run_btn:
    if not tickers:
        st.error("Please enter at least one stock ticker.")
    else:
        try:
            with st.spinner("Fetching data..."):
                raw_data, bench_raw = get_stock_data(tickers, benchmark, start_date, end_date)
            
            if raw_data.empty or bench_raw.empty:
                st.error("❌ No data found. Please check your tickers and date range.")
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
                st.error("❌ No overlapping data found between portfolio and benchmark.")
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
            
            # --- RESULTS DISPLAY ---
            st.success("✅ Analysis Complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Portfolio Return", f"{total_return:.2%}", 
                       delta=f"{(total_return - bench_total_return):.2%} vs Benchmark")
            col2.metric("Benchmark Return", f"{bench_total_return:.2%}")
            
            # Calculate volatility
            portfolio_vol = portfolio_daily.std() * (252 ** 0.5)  # Annualized
            col3.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
            
            # Chart
            st.subheader("📊 Growth Chart ($1 Investment)")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(portfolio_cum.index, portfolio_cum, label='Your Portfolio', linewidth=2.5, color='#1f77b4')
            ax.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark})', linestyle='--', linewidth=2, color='#ff7f0e')
            ax.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='Initial Investment')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title(f'Portfolio Performance: {start_date.date()} to {end_date.date()}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional stats in expander
            with st.expander("📈 See Detailed Statistics"):
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("**Portfolio Statistics**")
                    st.write(f"• Total Return: {total_return:.2%}")
                    st.write(f"• Annualized Volatility: {portfolio_vol:.2%}")
                    st.write(f"• Best Day: {portfolio_daily.max():.2%}")
                    st.write(f"• Worst Day: {portfolio_daily.min():.2%}")
                
                with stats_col2:
                    st.markdown("**Benchmark Statistics**")
                    bench_vol = bench_returns.std() * (252 ** 0.5)
                    st.write(f"• Total Return: {bench_total_return:.2%}")
                    st.write(f"• Annualized Volatility: {bench_vol:.2%}")
                    st.write(f"• Best Day: {bench_returns.max():.2%}")
                    st.write(f"• Worst Day: {bench_returns.min():.2%}")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.info("💡 Tip: Make sure all ticker symbols are valid and try a different date range.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    📱 Having issues on mobile? Use the <strong>Text Input</strong> date mode above.<br>
    Built with ❤️ using Streamlit & yfinance
    </div>
    """,
    unsafe_allow_html=True
)
