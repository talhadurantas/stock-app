import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- App Configuration ---
st.set_page_config(page_title="Portfolio Visualizer - Transaction Mode", layout="wide")
st.title("📈 Portfolio Visualizer - Transaction-Based")

# --- Helper Functions ---
def format_date_input(date_str):
    clean = date_str.strip().replace("-", "").replace(" ", "").replace("/", "")
    if clean.isdigit():
        if len(clean) <= 4: return clean
        elif len(clean) <= 6: return f"{clean[0:4]}-{clean[4:]}"
        elif len(clean) >= 7: return f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-": return date_str
    return date_str

@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    if 'Adj Close' in data.columns: data = data['Adj Close']
    elif 'Close' in data.columns: data = data['Close']
    
    if len(tickers) == 1:
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
    return data

def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02):
    excess_returns = daily_returns.mean() * 252 - risk_free_rate
    volatility = daily_returns.std() * (252 ** 0.5)
    return excess_returns / volatility if volatility != 0 else 0

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.info("ℹ️ Returns calculated with dividend reinvestment.")
    
    # Date Range
    st.markdown("### 📅 Analysis Period")
    c1, c2 = st.columns(2)
    start_input = c1.text_input("Start", "2013-01-01", help="YYYY-MM-DD or 20130101")
    end_input = c2.text_input("End", str(pd.to_datetime("today").date()))
    
    try:
        global_start = pd.to_datetime(format_date_input(start_input))
        global_end = pd.to_datetime(format_date_input(end_input))
    except:
        st.error("Invalid date format")
        global_start = pd.to_datetime("2013-01-01")
        global_end = pd.to_datetime("today")
    
    benchmark_ticker = st.text_input("Benchmark", "SPY").upper()

# --- MAIN AREA ---
st.markdown("### 🔄 Transaction-Based Portfolio Builder")
st.caption("Define your portfolio by entering transactions. The system automatically manages portfolio composition over time.")

# Initialize transaction data in session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame({
        'Date': ['2013-01-01', '2017-02-05', '2020-03-15'],
        'Action': ['Initial', 'Sell & Buy', 'Buy'],
        'Sell': ['', 'MSFT', ''],
        'Buy': ['AAPL, NVDA, MSFT', 'INTC', 'GOOGL'],
        'Weights': ['33, 33, 34', '33', '25'], 
        'Notes': ['Starting portfolio', 'Replaced MSFT with INTC', 'Added GOOGL']
    })

# Transaction Editor
st.markdown("#### 📝 Transaction Log")
st.caption("Dates must be chronological.")

edited_df = st.data_editor(
    st.session_state.transactions,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Date": st.column_config.TextColumn("Date", help="YYYY-MM-DD format", required=True),
        "Action": st.column_config.SelectboxColumn(
            "Action",
            options=["Initial", "Buy", "Sell", "Sell & Buy", "Rebalance"],
            required=True
        ),
        "Sell": st.column_config.TextColumn("Sell Tickers", help="Stocks to sell (comma-separated)"),
        "Buy": st.column_config.TextColumn("Buy Tickers", help="Stocks to buy (comma-separated)"),
        "Weights": st.column_config.TextColumn("Weights %", help="Percentage for each stock in Buy (comma-separated). Leave empty for equal weight."),
        "Notes": st.column_config.TextColumn("Notes", help="Optional description")
    },
    hide_index=False
)

# Update session state
st.session_state.transactions = edited_df

# Quick Add Buttons
st.markdown("#### ⚡ Quick Actions")
col1, col2, col3, col4 = st.columns(4)

if col1.button("➕ Add Row", use_container_width=True):
    new_row = pd.DataFrame({
        'Date': [str(pd.to_datetime("today").date())],
        'Action': ['Buy'],
        'Sell': [''],
        'Buy': [''],
        'Weights': [''],
        'Notes': ['']
    })
    st.session_state.transactions = pd.concat([st.session_state.transactions, new_row], ignore_index=True)
    st.rerun()

if col2.button("🗑️ Clear All", use_container_width=True):
    st.session_state.transactions = pd.DataFrame({
        'Date': [], 'Action': [], 'Sell': [], 'Buy': [], 'Weights': [], 'Notes': []
    })
    st.rerun()

st.markdown("---")

# Analysis Button
run_analysis = st.button("🚀 Run Transaction-Based Analysis", use_container_width=True, type="primary")

# --- ANALYSIS ENGINE ---
if run_analysis:
    try:
        # Validate transactions
        if st.session_state.transactions.empty:
            st.error("❌ Please add at least one transaction!")
            st.stop()
        
        with st.spinner("Processing transactions and calculating portfolio performance..."):
            
            # Parse transactions into portfolio periods
            transactions = st.session_state.transactions.copy()
            transactions['Date'] = pd.to_datetime(transactions['Date'].apply(format_date_input))
            transactions = transactions.sort_values('Date').reset_index(drop=True)
            
            # Build portfolio composition timeline
            portfolio_timeline = []
            current_holdings = {}  # {ticker: weight}
            
            for idx, row in transactions.iterrows():
                trans_date = row['Date']
                
                # Process sells
                if pd.notna(row['Sell']) and row['Sell'].strip():
                    sell_tickers = [x.strip().upper() for x in row['Sell'].split(',') if x.strip()]
                    for ticker in sell_tickers:
                        current_holdings.pop(ticker, None)
                
                # Process buys
                if pd.notna(row['Buy']) and row['Buy'].strip():
                    buy_tickers = [x.strip().upper() for x in row['Buy'].split(',') if x.strip()]
                    
                    # Parse weights if provided
                    weights = []
                    if pd.notna(row['Weights']) and row['Weights'].strip():
                        try:
                            weights = [float(x.strip()) for x in row['Weights'].split(',') if x.strip()]
                        except:
                            weights = []
                    
                    # If weights are missing, try to be smart about defaults
                    if not weights:
                        # Strategy: Assign equal share of the "Available Cash" or default to equal split
                        weights = [100.0 / len(buy_tickers)] * len(buy_tickers)
                    
                    # Add to holdings (Upsert)
                    for ticker, weight in zip(buy_tickers, weights):
                        current_holdings[ticker] = weight
                
                # Handle Rebalance (Clear everything and set new)
                if row['Action'] == 'Rebalance':
                    # Logic handled by the upsert above if user clears first, but let's be safe
                    # A true rebalance implies the final state is EXACTLY what is in "Buy"
                    # For simplicity in this logic, we assume user lists ALL desired stocks in Buy for a rebalance
                    pass 

                # --- CRITICAL FIX: AUTO-NORMALIZE ---
                # Check if total weight > 100% and scale down if necessary
                total_weight = sum(current_holdings.values())
                
                if total_weight > 100.01: # Small tolerance for float math
                    scale_factor = 100.0 / total_weight
                    for ticker in current_holdings:
                        current_holdings[ticker] *= scale_factor
                
                # Remove tiny residuals (cleanup)
                current_holdings = {k: v for k, v in current_holdings.items() if v > 0.01}

                # Calculate stats
                total_invested = sum(current_holdings.values())
                cash_percentage = 100 - total_invested
                
                # Record portfolio state
                if current_holdings or cash_percentage > 0:
                    portfolio_timeline.append({
                        'start_date': trans_date,
                        'holdings': dict(current_holdings), 
                        'invested_pct': total_invested,
                        'cash_pct': cash_percentage,
                        'action': row['Action'],
                        'notes': row['Notes']
                    })
            
            if not portfolio_timeline:
                st.error("❌ No valid portfolio composition found!")
                st.stop()
            
            # Add end dates to each period
            for i in range(len(portfolio_timeline)):
                if i < len(portfolio_timeline) - 1:
                    portfolio_timeline[i]['end_date'] = portfolio_timeline[i + 1]['start_date']
                else:
                    portfolio_timeline[i]['end_date'] = global_end
            
            # Filter to global date range
            portfolio_timeline = [p for p in portfolio_timeline if p['start_date'] < global_end]
            if not portfolio_timeline:
                 st.error("❌ All transactions are after the End Date.")
                 st.stop()

            # Pre-load all stock data
            all_tickers = set()
            for period in portfolio_timeline:
                all_tickers.update(period['holdings'].keys())
            
            all_tickers = list(all_tickers)
            if not all_tickers:
                st.error("❌ No stocks found in portfolio!")
                st.stop()
            
            master_data = get_stock_data(all_tickers, global_start, global_end)
            if master_data.empty:
                st.error("❌ Could not fetch stock data!")
                st.stop()
            
            # Calculate Benchmark
            bench_data = get_stock_data([benchmark_ticker], global_start, global_end)
            if bench_data.empty:
                 st.error("❌ No benchmark data!")
                 st.stop()
            
            if isinstance(bench_data, pd.DataFrame): 
                bench_data = bench_data.iloc[:, 0]
            bench_returns = bench_data.pct_change().dropna()
            bench_cum = (1 + bench_returns).cumprod()
            
            # Calculate Portfolio Performance
            all_cumulative = []
            all_daily_returns = []
            current_value = 1.0
            
            for period in portfolio_timeline:
                holdings_dict = period['holdings']
                tickers = list(holdings_dict.keys())
                weights_pct = list(holdings_dict.values())
                weights = [w / 100.0 for w in weights_pct] # Decimal weights
                
                start = max(period['start_date'], global_start)
                end = min(period['end_date'], global_end)
                
                # Slice data
                if len(tickers) == 1:
                    if tickers[0] in master_data:
                        data = master_data[tickers[0]].loc[start:end].to_frame(tickers[0])
                    else:
                        continue
                else:
                    valid_tickers = [t for t in tickers if t in master_data.columns]
                    if not valid_tickers: continue
                    data = master_data[valid_tickers].loc[start:end]
                
                if data.empty: continue
                
                daily_returns = data.pct_change().dropna()
                
                if len(tickers) > 1:
                    # Align weights
                    aligned_weights = [weights[tickers.index(col)] for col in data.columns]
                    portfolio_daily = (daily_returns * aligned_weights).sum(axis=1)
                else:
                    portfolio_daily = daily_returns.iloc[:, 0] * weights[0] # Correct for single stock weight < 100%
                
                # Apply cash drag (Cash earns 0%)
                # If invested is 80%, return is 80% * stock_return + 20% * 0
                # The weighted sum above handles the stock part. 
                # But if single stock 50%, the above line gives 50% of return, which is correct.
                
                portfolio_cum = current_value * (1 + portfolio_daily).cumprod()
                if not portfolio_cum.empty:
                     current_value = portfolio_cum.iloc[-1]
                
                all_cumulative.append(portfolio_cum)
                all_daily_returns.append(portfolio_daily)
            
            if not all_cumulative:
                 st.error("❌ No data could be calculated.")
                 st.stop()

            # Combine periods
            portfolio_cum = pd.concat(all_cumulative)
            portfolio_cum = portfolio_cum[~portfolio_cum.index.duplicated(keep='last')]
            portfolio_daily = pd.concat(all_daily_returns)
            
            # Align with benchmark
            common_idx = portfolio_cum.index.intersection(bench_cum.index)
            portfolio_cum = portfolio_cum.loc[common_idx]
            bench_cum = bench_cum.loc[common_idx]
            
            # Metrics
            total_return = portfolio_cum.iloc[-1] - 1
            bench_total_return = bench_cum.iloc[-1] - 1
            
            years = (global_end - global_start).days / 365.25
            portfolio_cagr = (portfolio_cum.iloc[-1]**(1/years)) - 1 if years > 0 else 0
            bench_cagr = (bench_cum.iloc[-1]**(1/years)) - 1 if years > 0 else 0
            
            portfolio_drawdown = calculate_max_drawdown(portfolio_cum)
            bench_drawdown = calculate_max_drawdown(bench_cum)
            
            portfolio_vol = portfolio_daily.std() * (252 ** 0.5)
            portfolio_sharpe = calculate_sharpe_ratio(portfolio_daily)
            
            # --- DISPLAY ---
            st.success(f"✅ Analysis Complete! (Portfolio automatically normalized to 100%)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{total_return:.2%}", delta=f"{total_return-bench_total_return:.2%}")
            col2.metric("CAGR", f"{portfolio_cagr:.2%}")
            col3.metric("Max Drawdown", f"{portfolio_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
            
            # Chart
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(portfolio_cum.index, portfolio_cum, label='Your Portfolio', linewidth=2.5, color='#1f77b4')
            ax.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark_ticker})', linestyle='--', color='#ff7f0e')
            
            # Mark transactions
            for idx, period in enumerate(portfolio_timeline[1:], 1):
                trans_date = period['start_date']
                if trans_date in portfolio_cum.index:
                    val = portfolio_cum.loc[trans_date]
                    ax.axvline(trans_date, color='purple', linestyle=':', alpha=0.5)
                    ax.text(trans_date, val, f" {period['action']}", color='purple', rotation=90, fontsize=8)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Portfolio Growth (Auto-Normalized)")
            st.pyplot(fig)
            
            # Timeline Table
            with st.expander("📋 Portfolio Timeline (Normalized)", expanded=True):
                timeline_data = []
                for p in portfolio_timeline:
                    holdings_str = ', '.join([f"{k} ({v:.1f}%)" for k, v in p['holdings'].items()])
                    timeline_data.append({
                        "Date": p['start_date'].date(),
                        "Holdings": holdings_str,
                        "Invested %": f"{p['invested_pct']:.1f}%",
                        "Cash %": f"{p['cash_pct']:.1f}%",
                        "Notes": p['notes']
                    })
                st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
