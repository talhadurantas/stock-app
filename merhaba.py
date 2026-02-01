import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# APP CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Portfolio Visualizer - Transaction Mode", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Professional Portfolio Visualizer - Transaction-Based")
st.caption("Track your portfolio performance with multi-period rebalancing and comprehensive analytics")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_date_input(date_str):
    """
    Automatically formats date input with real-time dash insertion.
    Accepts: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, YYYY MM DD
    Returns: YYYY-MM-DD format
    """
    clean = date_str.strip().replace("-", "").replace(" ", "").replace("/", "")
    
    if clean.isdigit():
        if len(clean) <= 4:
            return clean
        elif len(clean) <= 6:
            return f"{clean[0:4]}-{clean[4:]}"
        elif len(clean) >= 7:
            return f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
    
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str
    
    return date_str

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance with caching.
    Returns adjusted close prices.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            auto_adjust=False,
            progress=False
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Extract Adj Close
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif 'Close' in data.columns:
            data = data['Close']
        
        # Handle single ticker case
        if len(tickers) == 1 and isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown from peak.
    Returns: float (negative value representing % decline)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Find the peak and trough dates
    peak_idx = (cumulative_returns[:drawdown.idxmin()]).idxmax()
    trough_idx = drawdown.idxmin()
    
    return max_drawdown, peak_idx, trough_idx

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe Ratio.
    Default risk-free rate: 2% annually
    """
    if daily_returns.empty or daily_returns.std() == 0:
        return 0
    
    excess_returns = daily_returns.mean() * 252 - risk_free_rate
    volatility = daily_returns.std() * (252 ** 0.5)
    sharpe = excess_returns / volatility if volatility != 0 else 0
    
    return sharpe

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.02):
    """
    Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility).
    """
    if daily_returns.empty:
        return 0
    
    excess_returns = daily_returns.mean() * 252 - risk_free_rate
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * (252 ** 0.5)
    
    sortino = excess_returns / downside_std if downside_std != 0 else 0
    
    return sortino

def calculate_calmar_ratio(cagr, max_drawdown):
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).
    Measures return per unit of downside risk.
    """
    if max_drawdown == 0:
        return 0
    return cagr / abs(max_drawdown)

def validate_transaction_row(row, idx):
    """
    Validate a single transaction row.
    Returns: (is_valid: bool, error_message: str or None)
    """
    action = row['Action']
    sell = str(row['Sell']) if pd.notna(row['Sell']) else ""
    buy = str(row['Buy']) if pd.notna(row['Buy']) else ""
    date_str = str(row['Date']) if pd.notna(row['Date']) else ""
    
    # Validate date format
    try:
        pd.to_datetime(format_date_input(date_str))
    except:
        return False, f"Row {idx}: Invalid date format"
    
    # Validate action-specific requirements
    if action == "Sell & Buy":
        if not sell.strip() or not buy.strip():
            return False, f"Row {idx}: 'Sell & Buy' requires both Sell and Buy tickers"
    
    elif action == "Sell":
        if not sell.strip():
            return False, f"Row {idx}: 'Sell' requires Sell tickers"
    
    elif action in ["Initial", "Buy", "Rebalance"]:
        if not buy.strip():
            return False, f"Row {idx}: '{action}' requires Buy tickers"
    
    # Validate weights if provided
    if pd.notna(row['Weights']) and str(row['Weights']).strip():
        try:
            weights = [float(w.strip()) for w in str(row['Weights']).split(',') if w.strip()]
            if buy.strip():
                buy_tickers = [t.strip() for t in buy.split(',') if t.strip()]
                if len(weights) != len(buy_tickers):
                    return False, f"Row {idx}: Number of weights must match number of Buy tickers"
        except:
            return False, f"Row {idx}: Invalid weights format (use comma-separated numbers)"
    
    return True, None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("ℹ️ **Note:** Returns calculated with dividend reinvestment.")
    
    st.markdown("---")
    st.markdown("### 📅 Analysis Period")
    
    col_start, col_end = st.columns(2)
    
    with col_start:
        start_input = st.text_input(
            "Start Date", 
            "2013-01-01",
            help="Format: YYYY-MM-DD or YYYYMMDD"
        )
        start_formatted = format_date_input(start_input)
        if start_formatted != start_input and len(start_input) > 0:
            st.caption(f"→ {start_formatted}")
    
    with col_end:
        end_input = st.text_input(
            "End Date", 
            str(pd.to_datetime("today").date()),
            help="Format: YYYY-MM-DD or YYYYMMDD"
        )
        end_formatted = format_date_input(end_input)
        if end_formatted != end_input and len(end_input) > 0:
            st.caption(f"→ {end_formatted}")
    
    # Parse dates
    try:
        global_start = pd.to_datetime(start_formatted)
        global_end = pd.to_datetime(end_formatted)
        
        if global_start >= global_end:
            st.error("⚠️ Start date must be before end date!")
            global_start = pd.to_datetime("2013-01-01")
            global_end = pd.to_datetime("today")
    except:
        st.error("❌ Invalid date format!")
        global_start = pd.to_datetime("2013-01-01")
        global_end = pd.to_datetime("today")
    
    st.markdown("---")
    st.markdown("### 📊 Benchmark")
    
    benchmark_ticker = st.text_input(
        "Benchmark Ticker", 
        "SPY",
        help="Compare your portfolio against a benchmark (e.g., SPY, QQQ, ^GSPC)"
    ).upper()
    
    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    
    show_transaction_markers = st.checkbox("Show transaction markers on chart", value=True)
    show_drawdown_period = st.checkbox("Highlight max drawdown period", value=False)

# ============================================================================
# MAIN CONTENT - TRANSACTION TABLE
# ============================================================================

st.markdown("### 🔄 Transaction-Based Portfolio Builder")
st.caption("Define your portfolio evolution through transactions. The system automatically calculates composition and performance over time.")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame({
        'Date': ['2013-01-01', '2017-02-05', '2020-03-15'],
        'Action': ['Initial', 'Sell & Buy', 'Buy'],
        'Sell': ['', 'MSFT', ''],
        'Buy': ['AAPL, NVDA, MSFT', 'INTC', 'GOOGL'],
        'Weights': ['33, 33, 34', '33', '25'],
        'Notes': ['Starting portfolio', 'Replaced MSFT with INTC', 'Added GOOGL at 25%']
    })

# Information box about input rules
with st.expander("💡 Input Rules & Guidelines", expanded=False):
    st.markdown("""
    **Action Types:**
    - **Initial**: Start your portfolio with initial holdings
    - **Buy**: Add new stocks to your existing portfolio
    - **Sell**: Remove stocks from your portfolio (creates cash position)
    - **Sell & Buy**: Swap one or more stocks for others (requires both Sell and Buy)
    - **Rebalance**: Replace entire portfolio with new composition
    
    **Field Requirements:**
    - **Date**: Use YYYY-MM-DD format (e.g., 2013-01-01) or 8 digits (20130101)
    - **Sell**: Required for 'Sell' and 'Sell & Buy' actions
    - **Buy**: Required for all actions except 'Sell'
    - **Weights**: Optional. Comma-separated percentages for Buy tickers. Leave empty for equal weight.
    
    **Weight Examples:**
    - `40, 40, 20` = 40% first stock, 40% second, 20% third
    - Leave empty = Equal weight (e.g., 3 stocks = 33.33% each)
    - System auto-normalizes (e.g., `2, 2, 1` becomes 40%, 40%, 20%)
    """)

st.markdown("---")

# Transaction table editor
st.markdown("#### 📝 Transaction Log")

edited_df = st.data_editor(
    st.session_state.transactions,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Date": st.column_config.TextColumn(
            "Date",
            help="Transaction date (YYYY-MM-DD)",
            required=True,
            width="medium"
        ),
        "Action": st.column_config.SelectboxColumn(
            "Action",
            options=["Initial", "Buy", "Sell", "Sell & Buy", "Rebalance"],
            required=True,
            help="Type of transaction",
            width="medium"
        ),
        "Sell": st.column_config.TextColumn(
            "Sell Tickers",
            help="Stocks to sell (comma-separated)",
            width="medium"
        ),
        "Buy": st.column_config.TextColumn(
            "Buy Tickers",
            help="Stocks to buy (comma-separated)",
            width="medium"
        ),
        "Weights": st.column_config.TextColumn(
            "Weights %",
            help="Optional: Percentages for Buy tickers (comma-separated)",
            width="small"
        ),
        "Notes": st.column_config.TextColumn(
            "Notes",
            help="Optional description",
            width="large"
        )
    },
    hide_index=False
)

# Update session state
st.session_state.transactions = edited_df

# ============================================================================
# VALIDATION SYSTEM
# ============================================================================

validation_issues = []
invalid_rows = set()
transaction_valid = True

for idx, row in edited_df.iterrows():
    is_valid, error_msg = validate_transaction_row(row, idx)
    if not is_valid:
        validation_issues.append(error_msg)
        invalid_rows.add(idx)
        transaction_valid = False

# Display validation warnings with highlighted rows
if invalid_rows:
    st.markdown("#### ⚠️ Validation Issues Detected")
    
    def highlight_invalid_rows(row):
        if row.name in invalid_rows:
            return ['background-color: #fff3cd'] * len(row)  # Amber highlight
        return [''] * len(row)
    
    st.dataframe(
        edited_df.style.apply(highlight_invalid_rows, axis=1),
        use_container_width=True
    )
    
    st.warning("⚠️ **Input Validation Issues**")
    for issue in validation_issues:
        st.caption(f"• {issue}")
    st.caption("💡 Fix these issues to enable analysis. Problematic rows are highlighted in amber.")

# ============================================================================
# QUICK ACTIONS
# ============================================================================

st.markdown("---")
st.markdown("#### ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("➕ Add Row", use_container_width=True):
        new_row = pd.DataFrame({
            'Date': [str(pd.to_datetime("today").date())],
            'Action': ['Buy'],
            'Sell': [''],
            'Buy': [''],
            'Weights': [''],
            'Notes': ['']
        })
        st.session_state.transactions = pd.concat(
            [st.session_state.transactions, new_row], 
            ignore_index=True
        )
        st.rerun()

with col2:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.transactions = pd.DataFrame({
            'Date': [], 'Action': [], 'Sell': [], 'Buy': [], 'Weights': [], 'Notes': []
        })
        st.rerun()

with col3:
    if st.button("📋 Example: Tech Growth", use_container_width=True):
        st.session_state.transactions = pd.DataFrame({
            'Date': ['2013-01-01', '2017-02-05', '2020-03-15', '2022-01-10'],
            'Action': ['Initial', 'Sell & Buy', 'Buy', 'Rebalance'],
            'Sell': ['', 'MSFT', '', ''],
            'Buy': ['AAPL, NVDA, MSFT', 'INTC', 'GOOGL', 'AAPL, NVDA, GOOGL, TSLA'],
            'Weights': ['40, 40, 20', '33', '20', '25, 25, 25, 25'],
            'Notes': ['Start: Heavy AAPL/NVDA', 'Swap MSFT→INTC', 'Add GOOGL', 'Equal 4 stocks']
        })
        st.rerun()

with col4:
    if st.button("📋 Example: Value Play", use_container_width=True):
        st.session_state.transactions = pd.DataFrame({
            'Date': ['2013-01-01', '2018-06-15', '2021-09-20'],
            'Action': ['Initial', 'Buy', 'Sell & Buy'],
            'Sell': ['', '', 'BRK.B'],
            'Buy': ['BRK.B, JPM, JNJ', 'V', 'META'],
            'Weights': ['50, 25, 25', '20', '30'],
            'Notes': ['Value: 50% Berkshire', 'Add Visa', 'Tech pivot: Meta']
        })
        st.rerun()

st.markdown("---")

# ============================================================================
# RUN ANALYSIS BUTTON
# ============================================================================

run_analysis = st.button(
    "🚀 Run Transaction-Based Analysis",
    use_container_width=True,
    type="primary",
    disabled=not transaction_valid,
    help="Fix validation issues above to enable" if not transaction_valid else "Analyze portfolio performance with multi-period tracking"
)

# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

if run_analysis:
    try:
        with st.spinner("🔄 Processing transactions and fetching market data..."):
            
            # ================================================================
            # STEP 1: Parse and validate transactions
            # ================================================================
            
            transactions = edited_df.copy()
            transactions['Date'] = pd.to_datetime(transactions['Date'].apply(format_date_input))
            transactions = transactions.sort_values('Date').reset_index(drop=True)
            
            # ================================================================
            # STEP 2: Build portfolio composition timeline
            # ================================================================
            
            portfolio_timeline = []
            current_holdings = {}
            
            for idx, row in transactions.iterrows():
                trans_date = row['Date']
                action = row['Action']
                
                # Handle REBALANCE - reset holdings first
                if action == "Rebalance":
                    current_holdings = {}
                
                # Process SELLS
                if action != "Rebalance" and pd.notna(row['Sell']) and str(row['Sell']).strip():
                    sell_tickers = [t.strip().upper() for t in str(row['Sell']).split(',') if t.strip()]
                    for ticker in sell_tickers:
                        current_holdings.pop(ticker, None)
                
                # Process BUYS with custom weights
                if pd.notna(row['Buy']) and str(row['Buy']).strip():
                    buy_tickers = [t.strip().upper() for t in str(row['Buy']).split(',') if t.strip()]
                    
                    # Parse weights
                    weights = []
                    if pd.notna(row['Weights']) and str(row['Weights']).strip():
                        try:
                            weights = [float(w.strip()) for w in str(row['Weights']).split(',') if w.strip()]
                            # Normalize to sum to 100
                            total_weight = sum(weights)
                            if total_weight > 0:
                                weights = [w / total_weight * 100 for w in weights]
                        except:
                            weights = []
                    
                    # Use equal weights if not provided or incorrect number
                    if len(weights) != len(buy_tickers):
                        weights = [100.0 / len(buy_tickers)] * len(buy_tickers)
                    
                    # Add to holdings
                    for ticker, weight in zip(buy_tickers, weights):
                        current_holdings[ticker] = weight
                
                # Calculate total invested percentage
                total_invested = sum(current_holdings.values()) if current_holdings else 0
                cash_percentage = 100 - total_invested
                
                # Record portfolio state
                portfolio_timeline.append({
                    'start_date': trans_date,
                    'holdings': dict(current_holdings),
                    'invested_pct': total_invested,
                    'cash_pct': cash_percentage,
                    'action': action,
                    'notes': str(row['Notes']) if pd.notna(row['Notes']) else ''
                })
            
            if not portfolio_timeline:
                st.error("❌ No valid portfolio composition found!")
                st.stop()
            
            # ================================================================
            # STEP 3: Check for cash drag and show warnings
            # ================================================================
            
            periods_with_cash = [p for p in portfolio_timeline if p['cash_pct'] > 5]
            
            if periods_with_cash:
                st.warning(f"⚠️ **Cash Drag Detected!** {len(periods_with_cash)} period(s) have uninvested cash, which may reduce returns.")
                
                with st.expander("💡 View Cash Allocation Details", expanded=False):
                    for p in periods_with_cash:
                        st.write(f"**{p['start_date'].date()}**: {p['invested_pct']:.1f}% invested, **{p['cash_pct']:.1f}% cash**")
                        if p['cash_pct'] > 5:
                            st.caption(f"→ Consider adding a 'Buy' or 'Rebalance' transaction to reinvest the {p['cash_pct']:.1f}% cash")
            
            # Add end dates to each period
            for i in range(len(portfolio_timeline)):
                if i < len(portfolio_timeline) - 1:
                    portfolio_timeline[i]['end_date'] = portfolio_timeline[i + 1]['start_date']
                else:
                    portfolio_timeline[i]['end_date'] = global_end
            
            # Filter to global date range
            portfolio_timeline = [p for p in portfolio_timeline if p['start_date'] < global_end]
            
            if portfolio_timeline[0]['start_date'] > global_start:
                st.info(f"ℹ️ First transaction is after start date. Analysis begins from {portfolio_timeline[0]['start_date'].date()}")
            
            # ================================================================
            # STEP 4: Pre-load all stock data (OPTIMIZATION)
            # ================================================================
            
            # Collect all unique tickers
            all_tickers = set()
            for period in portfolio_timeline:
                all_tickers.update(period['holdings'].keys())
            
            all_tickers = sorted(list(all_tickers))
            
            if not all_tickers:
                st.error("❌ No stocks found in portfolio!")
                st.stop()
            
            # Download ALL data once
            with st.spinner(f"📥 Fetching data for {len(all_tickers)} stocks and benchmark..."):
                master_data = get_stock_data(all_tickers, global_start, global_end)
                bench_data = get_stock_data([benchmark_ticker], global_start, global_end)
            
            if master_data.empty:
                st.error("❌ Could not fetch stock data! Please check ticker symbols.")
                st.stop()
            
            if bench_data.empty:
                st.error(f"❌ Could not fetch benchmark data for {benchmark_ticker}!")
                st.stop()
            
            # Process benchmark
            if isinstance(bench_data, pd.DataFrame):
                bench_data = bench_data.iloc[:, 0]
            bench_returns = bench_data.pct_change().dropna()
            bench_cum = (1 + bench_returns).cumprod()
            
            # ================================================================
            # STEP 5: Calculate portfolio performance period by period
            # ================================================================
            
            all_cumulative = []
            all_daily_returns = []
            current_value = 1.0
            
            for period in portfolio_timeline:
                holdings_dict = period['holdings']
                
                if not holdings_dict:
                    continue
                
                tickers = list(holdings_dict.keys())
                weights_pct = list(holdings_dict.values())
                weights = [w / 100.0 for w in weights_pct]
                
                start = max(period['start_date'], global_start)
                end = min(period['end_date'], global_end)
                
                # Slice from pre-loaded master data
                try:
                    if len(tickers) == 1:
                        data = master_data[tickers[0]].loc[start:end].to_frame(tickers[0])
                    else:
                        data = master_data[tickers].loc[start:end]
                except:
                    continue
                
                if data.empty:
                    continue
                
                # Calculate returns
                daily_returns = data.pct_change().dropna()
                
                if len(tickers) > 1:
                    aligned_weights = [weights[tickers.index(col)] for col in data.columns]
                    portfolio_daily = (daily_returns * aligned_weights).sum(axis=1)
                else:
                    portfolio_daily = daily_returns.iloc[:, 0]
                
                # Calculate cumulative returns for this period
                portfolio_cum = current_value * (1 + portfolio_daily).cumprod()
                
                # Update current value for next period
                if not portfolio_cum.empty:
                    current_value = portfolio_cum.iloc[-1]
                
                all_cumulative.append(portfolio_cum)
                all_daily_returns.append(portfolio_daily)
            
            # ================================================================
            # STEP 6: Combine all periods
            # ================================================================
            
            portfolio_cum = pd.concat(all_cumulative)
            portfolio_cum = portfolio_cum[~portfolio_cum.index.duplicated(keep='last')]
            portfolio_daily = pd.concat(all_daily_returns)
            
            # Align portfolio and benchmark
            common_idx = portfolio_cum.index.intersection(bench_cum.index)
            portfolio_cum = portfolio_cum.loc[common_idx]
            bench_cum = bench_cum.loc[common_idx]
            portfolio_daily = portfolio_daily.loc[common_idx]
            bench_returns = bench_returns.loc[common_idx]
            
            # ================================================================
            # STEP 7: Calculate all metrics
            # ================================================================
            
            years = (global_end - global_start).days / 365.25
            
            # Returns
            total_return = portfolio_cum.iloc[-1] - 1
            bench_total_return = bench_cum.iloc[-1] - 1
            
            # CAGR
            portfolio_cagr = (portfolio_cum.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
            bench_cagr = (bench_cum.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
            
            # Volatility
            portfolio_vol = portfolio_daily.std() * (252 ** 0.5)
            bench_vol = bench_returns.std() * (252 ** 0.5)
            
            # Max Drawdown
            portfolio_drawdown, port_dd_peak, port_dd_trough = calculate_max_drawdown(portfolio_cum)
            bench_drawdown, bench_dd_peak, bench_dd_trough = calculate_max_drawdown(bench_cum)
            
            # Sharpe Ratio
            portfolio_sharpe = calculate_sharpe_ratio(portfolio_daily)
            bench_sharpe = calculate_sharpe_ratio(bench_returns)
            
            # Sortino Ratio
            portfolio_sortino = calculate_sortino_ratio(portfolio_daily)
            bench_sortino = calculate_sortino_ratio(bench_returns)
            
            # Calmar Ratio
            portfolio_calmar = calculate_calmar_ratio(portfolio_cagr, portfolio_drawdown)
            bench_calmar = calculate_calmar_ratio(bench_cagr, bench_drawdown)
            
            # Best/Worst Days
            portfolio_best_day = portfolio_daily.max()
            portfolio_worst_day = portfolio_daily.min()
            bench_best_day = bench_returns.max()
            bench_worst_day = bench_returns.min()
            
            # Win Rate
            portfolio_win_rate = (portfolio_daily > 0).sum() / len(portfolio_daily) * 100
            bench_win_rate = (bench_returns > 0).sum() / len(bench_returns) * 100
            
            # Average Invested Percentage
            avg_invested = sum([p['invested_pct'] for p in portfolio_timeline]) / len(portfolio_timeline)
            avg_cash = 100 - avg_invested
            
        # ================================================================
        # DISPLAY RESULTS
        # ================================================================
        
        st.success(f"✅ **Analysis Complete!** Analyzed {len(portfolio_timeline)} portfolio period(s) from {global_start.date()} to {global_end.date()}")
        
        st.markdown("---")
        st.markdown("### 📊 Performance Summary")
        
        # ================================================================
        # KEY METRICS - Row 1: Returns & CAGR
        # ================================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Portfolio Total Return",
            f"{total_return:.2%}",
            delta=f"{(total_return - bench_total_return):.2%} vs Benchmark",
            delta_color="normal"
        )
        
        col2.metric(
            "Portfolio CAGR",
            f"{portfolio_cagr:.2%}",
            help="Compound Annual Growth Rate"
        )
        
        col3.metric(
            "Benchmark Total Return",
            f"{bench_total_return:.2%}"
        )
        
        col4.metric(
            "Benchmark CAGR",
            f"{bench_cagr:.2%}",
            help="Compound Annual Growth Rate"
        )
        
        # ================================================================
        # KEY METRICS - Row 2: Risk Metrics
        # ================================================================
        
        col5, col6, col7, col8 = st.columns(4)
        
        col5.metric(
            "Portfolio Volatility",
            f"{portfolio_vol:.2%}",
            help="Annualized standard deviation"
        )
        
        col6.metric(
            "Portfolio Sharpe Ratio",
            f"{portfolio_sharpe:.2f}",
            help="Risk-adjusted return (>1 good, >2 very good)"
        )
        
        col7.metric(
            "Benchmark Volatility",
            f"{bench_vol:.2%}",
            help="Annualized standard deviation"
        )
        
        col8.metric(
            "Benchmark Sharpe Ratio",
            f"{bench_sharpe:.2f}",
            help="Risk-adjusted return"
        )
        
        # ================================================================
        # KEY METRICS - Row 3: Drawdown
        # ================================================================
        
        col9, col10 = st.columns(2)
        
        col9.metric(
            "Portfolio Max Drawdown",
            f"{portfolio_drawdown:.2%}",
            help=f"Peak: {port_dd_peak.date()}, Trough: {port_dd_trough.date()}"
        )
        
        col10.metric(
            "Benchmark Max Drawdown",
            f"{bench_drawdown:.2%}",
            help=f"Peak: {bench_dd_peak.date()}, Trough: {bench_dd_trough.date()}"
        )
        
        # ================================================================
        # VISUALIZATION: Growth Chart
        # ================================================================
        
        st.markdown("---")
        st.markdown("### 📈 Portfolio Growth Over Time")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot portfolio and benchmark
        ax.plot(
            portfolio_cum.index, 
            portfolio_cum, 
            label='Your Portfolio', 
            linewidth=2.5, 
            color='#1f77b4'
        )
        
        ax.plot(
            bench_cum.index, 
            bench_cum, 
            label=f'Benchmark ({benchmark_ticker})', 
            linestyle='--', 
            linewidth=2, 
            color='#ff7f0e'
        )
        
        # Add transaction markers
        if show_transaction_markers and len(portfolio_timeline) > 1:
            colors = ['purple', 'green', 'red', 'orange', 'brown', 'pink', 'gray', 'olive']
            for idx, period in enumerate(portfolio_timeline[1:], 1):
                color = colors[idx % len(colors)]
                trans_date = period['start_date']
                
                ax.axvline(
                    trans_date, 
                    color=color, 
                    linestyle=':', 
                    alpha=0.6, 
                    linewidth=1.5
                )
                
                # Add label
                y_position = portfolio_cum.max() * (0.95 - (idx % 5) * 0.05)
                ax.text(
                    trans_date, 
                    y_position, 
                    f' {period["action"]}', 
                    color=color, 
                    rotation=90, 
                    fontsize=8,
                    va='top'
                )
        
        # Highlight max drawdown period
        if show_drawdown_period:
            ax.axvspan(
                port_dd_peak, 
                port_dd_trough, 
                alpha=0.2, 
                color='red',
                label=f'Max Drawdown Period'
            )
        
        # Formatting
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Multi-Period Portfolio Performance: {global_start.date()} to {global_end.date()}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # ================================================================
        # PORTFOLIO TIMELINE TABLE
        # ================================================================
        
        st.markdown("---")
        
        with st.expander("📋 Portfolio Timeline & Cash Allocation", expanded=True):
            timeline_display = []
            
            for p in portfolio_timeline:
                holdings_dict = p['holdings']
                holdings_str = ', '.join([
                    f"{ticker} ({weight:.1f}%)" 
                    for ticker, weight in holdings_dict.items()
                ])
                
                timeline_display.append({
                    'Start Date': p['start_date'].date(),
                    'End Date': p['end_date'].date(),
                    'Holdings (Weight %)': holdings_str,
                    'Invested %': f"{p['invested_pct']:.1f}%",
                    'Cash %': f"{p['cash_pct']:.1f}%",
                    'Action': p['action'],
                    'Notes': p['notes']
                })
            
            df_timeline = pd.DataFrame(timeline_display)
            
            # Highlight rows with significant cash
            def highlight_cash_rows(row):
                try:
                    cash_val = float(row['Cash %'].rstrip('%'))
                    if cash_val > 5:
                        return ['background-color: #fff3cd'] * len(row)
                except:
                    pass
                return [''] * len(row)
            
            st.dataframe(
                df_timeline.style.apply(highlight_cash_rows, axis=1),
                use_container_width=True
            )
            
            st.caption("💡 Rows highlighted in amber have >5% uninvested cash. Consider rebalancing to improve returns.")
        
        # ================================================================
        # DETAILED STATISTICS
        # ================================================================
        
        with st.expander("📈 Detailed Performance Statistics", expanded=False):
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.markdown("#### 📊 Portfolio Statistics")
                st.write(f"**Returns:**")
                st.write(f"• Total Return: {total_return:.2%}")
                st.write(f"• CAGR: {portfolio_cagr:.2%}")
                st.write(f"• Best Day: {portfolio_best_day:.2%} ({portfolio_daily.idxmax().date()})")
                st.write(f"• Worst Day: {portfolio_worst_day:.2%} ({portfolio_daily.idxmin().date()})")
                st.write(f"• Win Rate: {portfolio_win_rate:.1f}%")
                
                st.write(f"")
                st.write(f"**Risk Metrics:**")
                st.write(f"• Volatility (Ann.): {portfolio_vol:.2%}")
                st.write(f"• Max Drawdown: {portfolio_drawdown:.2%}")
                st.write(f"• Drawdown Peak: {port_dd_peak.date()}")
                st.write(f"• Drawdown Trough: {port_dd_trough.date()}")
                st.write(f"• Drawdown Duration: {(port_dd_trough - port_dd_peak).days} days")
                
                st.write(f"")
                st.write(f"**Risk-Adjusted Returns:**")
                st.write(f"• Sharpe Ratio: {portfolio_sharpe:.2f}")
                st.write(f"• Sortino Ratio: {portfolio_sortino:.2f}")
                st.write(f"• Calmar Ratio: {portfolio_calmar:.2f}")
                
                st.write(f"")
                st.write(f"**Portfolio Info:**")
                st.write(f"• Time Period: {years:.2f} years")
                st.write(f"• Total Transactions: {len(transactions)}")
                st.write(f"• Portfolio Periods: {len(portfolio_timeline)}")
                st.write(f"• Unique Stocks: {len(all_tickers)}")
                st.write(f"• Avg Invested %: {avg_invested:.1f}%")
                if avg_cash > 5:
                    st.write(f"• Avg Cash %: {avg_cash:.1f}% ⚠️")
            
            with stats_col2:
                st.markdown("#### 📊 Benchmark Statistics")
                st.write(f"**Returns:**")
                st.write(f"• Total Return: {bench_total_return:.2%}")
                st.write(f"• CAGR: {bench_cagr:.2%}")
                st.write(f"• Best Day: {bench_best_day:.2%} ({bench_returns.idxmax().date()})")
                st.write(f"• Worst Day: {bench_worst_day:.2%} ({bench_returns.idxmin().date()})")
                st.write(f"• Win Rate: {bench_win_rate:.1f}%")
                
                st.write(f"")
                st.write(f"**Risk Metrics:**")
                st.write(f"• Volatility (Ann.): {bench_vol:.2%}")
                st.write(f"• Max Drawdown: {bench_drawdown:.2%}")
                st.write(f"• Drawdown Peak: {bench_dd_peak.date()}")
                st.write(f"• Drawdown Trough: {bench_dd_trough.date()}")
                st.write(f"• Drawdown Duration: {(bench_dd_trough - bench_dd_peak).days} days")
                
                st.write(f"")
                st.write(f"**Risk-Adjusted Returns:**")
                st.write(f"• Sharpe Ratio: {bench_sharpe:.2f}")
                st.write(f"• Sortino Ratio: {bench_sortino:.2f}")
                st.write(f"• Calmar Ratio: {bench_calmar:.2f}")
                
                st.write(f"")
                st.write(f"**Comparison vs Portfolio:**")
                excess_return = total_return - bench_total_return
                excess_cagr = portfolio_cagr - bench_cagr
                excess_sharpe = portfolio_sharpe - bench_sharpe
                
                st.write(f"• Excess Return: {excess_return:+.2%}")
                st.write(f"• Excess CAGR: {excess_cagr:+.2%}")
                st.write(f"• Excess Sharpe: {excess_sharpe:+.2f}")
                
                if excess_return > 0:
                    st.success("✅ Portfolio outperformed benchmark")
                else:
                    st.error("❌ Portfolio underperformed benchmark")
        
        # ================================================================
        # ROLLING METRICS
        # ================================================================
        
        with st.expander("📉 Rolling Metrics (1-Year Window)", expanded=False):
            
            # Calculate rolling metrics
            rolling_window = 252  # 1 year
            
            if len(portfolio_daily) > rolling_window:
                rolling_return = portfolio_cum.pct_change(rolling_window)
                rolling_vol = portfolio_daily.rolling(rolling_window).std() * (252 ** 0.5)
                rolling_sharpe = (
                    portfolio_daily.rolling(rolling_window).mean() * 252 / 
                    (portfolio_daily.rolling(rolling_window).std() * (252 ** 0.5))
                )
                
                fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
                
                # Rolling Return
                ax1.plot(rolling_return.index, rolling_return * 100, color='#1f77b4', linewidth=1.5)
                ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_title('Rolling 1-Year Return (%)', fontweight='bold')
                ax1.set_ylabel('Return (%)')
                ax1.grid(True, alpha=0.3)
                
                # Rolling Volatility
                ax2.plot(rolling_vol.index, rolling_vol * 100, color='#ff7f0e', linewidth=1.5)
                ax2.set_title('Rolling 1-Year Volatility (%)', fontweight='bold')
                ax2.set_ylabel('Volatility (%)')
                ax2.grid(True, alpha=0.3)
                
                # Rolling Sharpe
                ax3.plot(rolling_sharpe.index, rolling_sharpe, color='#2ca02c', linewidth=1.5)
                ax3.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 1')
                ax3.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 2')
                ax3.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.set_xlabel('Date')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("ℹ️ Not enough data for rolling metrics (need at least 1 year of daily data)")
        
    except Exception as e:
        st.error(f"❌ **Analysis Error:** {str(e)}")
        st.exception(e)
        st.info("💡 **Troubleshooting Tips:**\n- Check that all ticker symbols are valid\n- Ensure dates are in correct format\n- Verify that transactions are chronological\n- Make sure weights sum to reasonable values")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
    <strong>Professional Portfolio Visualizer - Transaction Mode</strong><br>
    Built with ❤️ using Streamlit & yfinance<br>
    Multi-period analysis with cash tracking & comprehensive risk metrics<br>
    <em>For educational purposes only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)
