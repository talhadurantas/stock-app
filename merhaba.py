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
        'Notes': ['Starting portfolio', 'Replaced MSFT with INTC', 'Added GOOGL at 25%']
    })

# Transaction Editor
st.markdown("#### 📝 Transaction Log")
st.caption("Edit the table below to define your portfolio changes. Dates must be chronological.")

# Add validation info box
st.info("💡 **Input Rules:** 'Initial' & 'Buy' require only Buy tickers. 'Sell & Buy' requires both Sell and Buy tickers. 'Rebalance' replaces entire portfolio.")

edited_df = st.data_editor(
    st.session_state.transactions,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Date": st.column_config.TextColumn("Date", help="YYYY-MM-DD format", required=True),
        "Action": st.column_config.SelectboxColumn(
            "Action",
            options=["Initial", "Buy", "Sell", "Sell & Buy", "Rebalance"],
            required=True,
            help="Initial: Start portfolio | Buy: Add stocks | Sell: Remove stocks | Sell & Buy: Swap stocks | Rebalance: Replace all"
        ),
        "Sell": st.column_config.TextColumn("Sell Tickers", help="Required for 'Sell' and 'Sell & Buy' actions (comma-separated)"),
        "Buy": st.column_config.TextColumn("Buy Tickers", help="Required for all actions except 'Sell' (comma-separated)"),
        "Weights": st.column_config.TextColumn("Weights %", help="Percentage for each Buy ticker (comma-separated). Leave empty for equal weight."),
        "Notes": st.column_config.TextColumn("Notes", help="Optional description")
    },
    hide_index=False
)

# Update session state
st.session_state.transactions = edited_df

# === VALIDATION SYSTEM ===
validation_issues = []
transaction_valid = True

# Validate each transaction
for idx, row in edited_df.iterrows():
    action = row['Action']
    sell = row['Sell'] if pd.notna(row['Sell']) else ""
    buy = row['Buy'] if pd.notna(row['Buy']) else ""
    
    # Check for action-specific requirements
    if action == "Sell & Buy":
        if not sell.strip() or not buy.strip():
            validation_issues.append(f"Row {idx}: 'Sell & Buy' requires both Sell and Buy tickers")
            transaction_valid = False
    
    elif action == "Sell":
        if not sell.strip():
            validation_issues.append(f"Row {idx}: 'Sell' requires Sell tickers")
            transaction_valid = False
    
    elif action in ["Initial", "Buy", "Rebalance"]:
        if not buy.strip():
            validation_issues.append(f"Row {idx}: '{action}' requires Buy tickers")
            transaction_valid = False
    
    # Check for date format
    try:
        pd.to_datetime(format_date_input(row['Date']))
    except:
        validation_issues.append(f"Row {idx}: Invalid date format")
        transaction_valid = False

# Display validation warnings (elegant, not errors)
if validation_issues:
    st.warning("⚠️ **Input Validation Issues**")
    for issue in validation_issues:
        st.caption(f"• {issue}")
    st.caption("💡 Fix these issues before running analysis")


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

if col3.button("📋 Example 1: Tech Growth", use_container_width=True):
    st.session_state.transactions = pd.DataFrame({
        'Date': ['2013-01-01', '2017-02-05', '2020-03-15', '2022-01-10'],
        'Action': ['Initial', 'Sell & Buy', 'Buy', 'Rebalance'],
        'Sell': ['', 'MSFT', '', ''],
        'Buy': ['AAPL, NVDA, MSFT', 'INTC', 'GOOGL', 'AAPL, NVDA, GOOGL, TSLA'],
        'Weights': ['40, 40, 20', '33', '20', '25, 25, 25, 25'],
        'Notes': ['Start: Heavy on AAPL/NVDA', 'Swap MSFT→INTC', 'Add GOOGL (20%)', 'Equal weight 4 stocks']
    })
    st.rerun()

if col4.button("📋 Example 2: Value Play", use_container_width=True):
    st.session_state.transactions = pd.DataFrame({
        'Date': ['2013-01-01', '2018-06-15', '2021-09-20'],
        'Action': ['Initial', 'Buy', 'Sell & Buy'],
        'Sell': ['', '', 'BRK.B'],
        'Buy': ['BRK.B, JPM, JNJ', 'V', 'META'],
        'Weights': ['50, 25, 25', '20', '30'],
        'Notes': ['Value stocks: 50% Berkshire', 'Add Visa 20%', 'Tech pivot: 30% Meta']
    })
    st.rerun()

st.markdown("---")

# Analysis Button - disabled if validation fails
run_analysis = st.button(
    "🚀 Run Transaction-Based Analysis", 
    use_container_width=True, 
    type="primary",
    disabled=not transaction_valid,
    help="Fix validation issues above to enable" if not transaction_valid else "Analyze portfolio performance"
)

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
            current_holdings = {}  # Changed to dict to store weights: {ticker: weight}
            
            for idx, row in transactions.iterrows():
                trans_date = row['Date']
                
                # Process sells
                if pd.notna(row['Sell']) and row['Sell'].strip():
                    sell_tickers = [x.strip().upper() for x in row['Sell'].split(',') if x.strip()]
                    for ticker in sell_tickers:
                        current_holdings.pop(ticker, None)
                
                # Process buys with custom weights
                if pd.notna(row['Buy']) and row['Buy'].strip():
                    buy_tickers = [x.strip().upper() for x in row['Buy'].split(',') if x.strip()]
                    
                    # Parse weights if provided
                    weights = []
                    if pd.notna(row['Weights']) and row['Weights'].strip():
                        try:
                            weights = [float(x.strip()) for x in row['Weights'].split(',') if x.strip()]
                            # Normalize weights to sum to 100
                            total_weight = sum(weights)
                            if total_weight > 0:
                                weights = [w / total_weight * 100 for w in weights]
                        except:
                            weights = []
                    
                    # If no weights or wrong number, use equal weights
                    if len(weights) != len(buy_tickers):
                        weights = [100.0 / len(buy_tickers)] * len(buy_tickers)
                    
                    # Add to holdings
                    for ticker, weight in zip(buy_tickers, weights):
                        current_holdings[ticker] = weight
                
                # Handle rebalance (replace all holdings with new weights)
                if row['Action'] == 'Rebalance' and pd.notna(row['Buy']) and row['Buy'].strip():
                    buy_tickers = [x.strip().upper() for x in row['Buy'].split(',') if x.strip()]
                    
                    # Parse weights
                    weights = []
                    if pd.notna(row['Weights']) and row['Weights'].strip():
                        try:
                            weights = [float(x.strip()) for x in row['Weights'].split(',') if x.strip()]
                            total_weight = sum(weights)
                            if total_weight > 0:
                                weights = [w / total_weight * 100 for w in weights]
                        except:
                            weights = []
                    
                    if len(weights) != len(buy_tickers):
                        weights = [100.0 / len(buy_tickers)] * len(buy_tickers)
                    
                    # Replace all holdings
                    current_holdings = {ticker: weight for ticker, weight in zip(buy_tickers, weights)}
                
                # Calculate total invested percentage
                total_invested = sum(current_holdings.values()) if current_holdings else 0
                cash_percentage = 100 - total_invested
                
                # Record portfolio state
                if current_holdings or cash_percentage > 0:
                    portfolio_timeline.append({
                        'start_date': trans_date,
                        'holdings': dict(current_holdings),  # Store as dict with weights
                        'invested_pct': total_invested,
                        'cash_pct': cash_percentage,
                        'action': row['Action'],
                        'notes': row['Notes']
                    })
            
            if not portfolio_timeline:
                st.error("❌ No valid portfolio composition found!")
                st.stop()
            
            # Check for cash drag and show warnings
            periods_with_cash = [p for p in portfolio_timeline if p['cash_pct'] > 5]
            if periods_with_cash:
                st.warning(f"⚠️ **Cash Drag Detected!** You have {len(periods_with_cash)} period(s) with uninvested cash. This will reduce returns.")
                with st.expander("💡 See Cash Allocation Details"):
                    for p in periods_with_cash:
                        st.write(f"**{p['start_date'].date()}**: {p['invested_pct']:.1f}% invested, {p['cash_pct']:.1f}% cash")
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
                st.warning(f"⚠️ First transaction is after start date. Analysis begins from {portfolio_timeline[0]['start_date'].date()}")
            
            # === OPTIMIZATION: Pre-load all stock data in ONE API call ===
            # Collect all unique tickers from all periods
            all_tickers = set()
            for period in portfolio_timeline:
                all_tickers.update(period['holdings'].keys())
            
            all_tickers = list(all_tickers)
            
            if not all_tickers:
                st.error("❌ No stocks found in portfolio!")
                st.stop()
            
            # Download ALL data once for the entire period
            with st.spinner(f"Fetching data for {len(all_tickers)} stocks..."):
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
            
            # Calculate Portfolio Performance (Period by Period)
            all_cumulative = []
            all_daily_returns = []
            current_value = 1.0
            
            for period in portfolio_timeline:
                holdings_dict = period['holdings']
                tickers = list(holdings_dict.keys())
                weights_pct = list(holdings_dict.values())
                
                # Convert percentages to decimals
                weights = [w / 100.0 for w in weights_pct]
                
                start = max(period['start_date'], global_start)
                end = min(period['end_date'], global_end)
                
                # === OPTIMIZED: Slice from pre-loaded master data ===
                if len(tickers) == 1:
                    data = master_data[tickers[0]].loc[start:end].to_frame(tickers[0])
                else:
                    data = master_data[tickers].loc[start:end]
                
                if data.empty:
                    continue
                
                daily_returns = data.pct_change().dropna()
                
                if len(tickers) > 1:
                    # Align weights with actual columns in data
                    aligned_weights = [weights[tickers.index(col)] for col in data.columns]
                    portfolio_daily = (daily_returns * aligned_weights).sum(axis=1)
                else:
                    portfolio_daily = daily_returns.iloc[:, 0]
                
                portfolio_cum = current_value * (1 + portfolio_daily).cumprod()
                current_value = portfolio_cum.iloc[-1] if not portfolio_cum.empty else current_value
                
                all_cumulative.append(portfolio_cum)
                all_daily_returns.append(portfolio_daily)
            
            # Combine all periods
            portfolio_cum = pd.concat(all_cumulative)
            portfolio_cum = portfolio_cum[~portfolio_cum.index.duplicated(keep='last')]
            portfolio_daily = pd.concat(all_daily_returns)
            
            # Align with benchmark
            common_idx = portfolio_cum.index.intersection(bench_cum.index)
            portfolio_cum = portfolio_cum.loc[common_idx]
            bench_cum = bench_cum.loc[common_idx]
            
            # Calculate metrics
            total_return = portfolio_cum.iloc[-1] - 1
            bench_total_return = bench_cum.iloc[-1] - 1
            
            years = (global_end - global_start).days / 365.25
            portfolio_cagr = (portfolio_cum.iloc[-1]**(1/years)) - 1 if years > 0 else 0
            bench_cagr = (bench_cum.iloc[-1]**(1/years)) - 1 if years > 0 else 0
            
            portfolio_drawdown = calculate_max_drawdown(portfolio_cum)
            bench_drawdown = calculate_max_drawdown(bench_cum)
            
            portfolio_sharpe = calculate_sharpe_ratio(portfolio_daily)
            bench_sharpe = calculate_sharpe_ratio(bench_returns)
            
            portfolio_vol = portfolio_daily.std() * (252 ** 0.5)
            bench_vol = bench_returns.std() * (252 ** 0.5)
            
            # --- DISPLAY RESULTS ---
            st.success(f"✅ Analysis Complete! ({len(portfolio_timeline)} portfolio periods)")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Return", f"{total_return:.2%}", 
                       delta=f"{(total_return - bench_total_return):.2%} vs Benchmark")
            col2.metric("Portfolio CAGR", f"{portfolio_cagr:.2%}")
            col3.metric("Benchmark Return", f"{bench_total_return:.2%}")
            col4.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
            col6.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}")
            col7.metric("Benchmark Volatility", f"{bench_vol:.2%}")
            col8.metric("Benchmark Sharpe", f"{bench_sharpe:.2f}")
            
            col9, col10 = st.columns(2)
            col9.metric("Portfolio Max Drawdown", f"{portfolio_drawdown:.2%}")
            col10.metric("Benchmark Max Drawdown", f"{bench_drawdown:.2%}")
            
            # Chart
            st.subheader("📊 Portfolio Growth Over Time")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(portfolio_cum.index, portfolio_cum, label='Your Portfolio', linewidth=2.5, color='#1f77b4')
            ax.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark_ticker})', 
                   linestyle='--', linewidth=2, color='#ff7f0e')
            
            # Mark transaction dates
            colors = ['purple', 'green', 'red', 'orange', 'brown', 'pink', 'gray']
            for idx, period in enumerate(portfolio_timeline[1:], 1):  # Skip first (initial)
                color = colors[idx % len(colors)]
                trans_date = period['start_date']
                ax.axvline(trans_date, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
                ax.text(trans_date, portfolio_cum.max() * (0.95 - idx*0.03), 
                       f' {period["action"]}', color=color, rotation=90, fontsize=8)
            
            ax.axhline(1.0, color='red', linestyle=':', alpha=0.3)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title(f'Transaction-Based Portfolio: {global_start.date()} to {global_end.date()}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Portfolio Timeline Table
            with st.expander("📋 Portfolio Timeline & Cash Allocation"):
                timeline_display = []
                for p in portfolio_timeline:
                    holdings_dict = p['holdings']
                    holdings_str = ', '.join([f"{ticker} ({weight:.1f}%)" for ticker, weight in holdings_dict.items()])
                    
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
                
                # Highlight rows with significant cash (elegant amber color)
                def highlight_cash(row):
                    cash_val = float(row['Cash %'].rstrip('%'))
                    if cash_val > 5:
                        return ['background-color: #fff3cd'] * len(row)  # Soft amber
                    return [''] * len(row)
                
                st.dataframe(df_timeline.style.apply(highlight_cash, axis=1), use_container_width=True)
                st.caption("💡 Rows highlighted in amber have >5% uninvested cash - consider rebalancing to improve returns")
            
            # Detailed Stats
            with st.expander("📈 Detailed Statistics"):
                stats_col1, stats_col2 = st.columns(2)
                
                # Calculate average invested percentage across all periods
                avg_invested = sum([p['invested_pct'] for p in portfolio_timeline]) / len(portfolio_timeline)
                
                with stats_col1:
                    st.markdown("**Portfolio Statistics**")
                    st.write(f"• Total Return: {total_return:.2%}")
                    st.write(f"• CAGR: {portfolio_cagr:.2%}")
                    st.write(f"• Volatility: {portfolio_vol:.2%}")
                    st.write(f"• Sharpe Ratio: {portfolio_sharpe:.2f}")
                    st.write(f"• Max Drawdown: {portfolio_drawdown:.2%}")
                    st.write(f"• Transactions: {len(transactions)}")
                    st.write(f"• **Avg Invested %: {avg_invested:.1f}%**")
                    if avg_invested < 95:
                        st.write(f"• **Avg Cash %: {100 - avg_invested:.1f}%** ⚠️")
                
                with stats_col2:
                    st.markdown("**Benchmark Statistics**")
                    st.write(f"• Total Return: {bench_total_return:.2%}")
                    st.write(f"• CAGR: {bench_cagr:.2%}")
                    st.write(f"• Volatility: {bench_vol:.2%}")
                    st.write(f"• Sharpe Ratio: {bench_sharpe:.2f}")
                    st.write(f"• Max Drawdown: {bench_drawdown:.2%}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Transaction-Based Portfolio Analysis | Built with Streamlit & yfinance
</div>
""", unsafe_allow_html=True)
