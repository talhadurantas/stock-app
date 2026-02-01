import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="Pro Portfolio Visualizer", layout="wide")
st.title("📈 Pro Stock Portfolio Visualizer (Multi-Stage)")

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

def calculate_segment_performance(tickers, weights, start_date, end_date, initial_value=1.0):
    data = get_stock_data(tickers, start_date, end_date)
    
    if data.empty:
        return None, None
    
    daily_returns = data.pct_change().dropna()
    
    if len(tickers) > 1:
        aligned_weights = [weights[tickers.index(col)] for col in data.columns]
        portfolio_daily = (daily_returns * aligned_weights).sum(axis=1)
    else:
        portfolio_daily = daily_returns.iloc[:, 0]
    
    portfolio_cum = initial_value * (1 + portfolio_daily).cumprod()
    
    return portfolio_cum, portfolio_daily

# --- Initialize Session State for Dynamic Periods ---
if 'num_periods' not in st.session_state:
    st.session_state.num_periods = 1

def add_period():
    st.session_state.num_periods += 1

def remove_period():
    if st.session_state.num_periods > 1:
        st.session_state.num_periods -= 1

# --- Sidebar ---
with st.sidebar:
    st.header("Portfolio Settings")
    
    st.info("ℹ️ **Note:** Returns calculated with dividend reinvestment.")
    
    # --- GLOBAL DATES ---
    st.markdown("### 📅 Overall Timeframe")
    
    date_mode = st.radio("Date Input:", ["Text Input (Type 8 digits)", "Dropdown Calendar (Select from lists)"], 
                         index=0, label_visibility="collapsed")
    
    if date_mode == "Text Input (Type 8 digits)":
        default_start = "2013-01-01"
        default_end = str(pd.to_datetime("today").date())
        
        c1, c2 = st.columns(2)
        start_input = c1.text_input("Start Date", default_start, placeholder="20150101")
        end_input = c2.text_input("End Date", default_end, placeholder="20250201")
        
        start_formatted = format_date_input(start_input)
        end_formatted = format_date_input(end_input)
        
        if start_formatted != start_input and len(start_input) > 0:
            c1.caption(f"→ {start_formatted}")
        if end_formatted != end_input and len(end_input) > 0:
            c2.caption(f"→ {end_formatted}")
        
        try:
            global_start = pd.to_datetime(start_formatted)
            global_end = pd.to_datetime(end_formatted)
            if global_start >= global_end:
                st.warning("⚠️ Start must be before End!")
        except:
            st.error("❌ Invalid date format!")
            global_start = pd.to_datetime("2013-01-01")
            global_end = pd.to_datetime("today")
    else:
        st.markdown("**📅 Select dates using dropdowns:**")
        st.write("**Start Date:**")
        col1, col2, col3 = st.columns(3)
        start_year = col1.selectbox("Year", range(2000, 2027), index=13, key="start_year")
        start_month = col2.selectbox("Month", range(1, 13), index=0, key="start_month")
        start_day = col3.selectbox("Day", range(1, 32), index=0, key="start_day")
        
        st.write("**End Date:**")
        col4, col5, col6 = st.columns(3)
        today = pd.to_datetime("today")
        end_year = col4.selectbox("Year", range(2000, 2027), index=26, key="end_year")
        end_month = col5.selectbox("Month", range(1, 13), index=today.month-1, key="end_month")
        end_day = col6.selectbox("Day", range(1, 32), index=today.day-1, key="end_day")
        
        try:
            global_start = pd.to_datetime(f"{start_year}-{start_month:02d}-{start_day:02d}")
            global_end = pd.to_datetime(f"{end_year}-{end_month:02d}-{end_day:02d}")
        except:
            st.error("❌ Invalid date!")
            global_start = pd.to_datetime("2013-01-01")
            global_end = pd.to_datetime("today")

    benchmark_ticker = st.text_input("Benchmark Ticker", "SPY").upper()

    st.markdown("---")

# --- MAIN CONTENT AREA ---
st.markdown("### 🔄 Portfolio Periods")
st.caption("Add multiple periods to simulate buying/selling stocks at different dates")

# Period management buttons
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 3])
with col_btn1:
    st.button("➕ Add Period", on_click=add_period, use_container_width=True)
with col_btn2:
    st.button("➖ Remove Period", on_click=remove_period, use_container_width=True, 
              disabled=(st.session_state.num_periods <= 1))
with col_btn3:
    st.caption(f"Total Periods: {st.session_state.num_periods}")

st.markdown("---")

# --- DYNAMIC FORM FOR MULTIPLE PERIODS ---
with st.form(key='multi_period_form'):
    
    periods_data = []
    
    for period_idx in range(st.session_state.num_periods):
        period_num = period_idx + 1
        
        # Visual separator
        if period_idx > 0:
            st.markdown("---")
        
        # Period Header
        if period_num == 1:
            st.subheader(f"📊 Period {period_num}: {global_start.date()} → {'Period 2 start' if st.session_state.num_periods > 1 else global_end.date()}")
        elif period_num == st.session_state.num_periods:
            st.subheader(f"📊 Period {period_num}: Rebalance Date → {global_end.date()}")
        else:
            st.subheader(f"📊 Period {period_num}: Rebalance Date → Next Rebalance")
        
        # Rebalance Date (only for periods 2+)
        rebalance_date = None
        if period_num > 1:
            if date_mode == "Text Input (Type 8 digits)":
                default_reb = f"201{6 + period_idx}-01-01"
                reb_input = st.text_input(
                    f"Rebalance Date for Period {period_num}", 
                    default_reb, 
                    key=f"reb_date_{period_num}",
                    help="Date when this portfolio composition starts"
                )
                reb_formatted = format_date_input(reb_input)
                if reb_formatted != reb_input and len(reb_input) > 0:
                    st.caption(f"→ {reb_formatted}")
                try:
                    rebalance_date = pd.to_datetime(reb_formatted)
                except:
                    st.error(f"Invalid rebalance date for Period {period_num}")
                    rebalance_date = global_start
            else:
                rebalance_date = st.date_input(
                    f"Rebalance Date for Period {period_num}",
                    pd.to_datetime(f"201{6 + period_idx}-01-01"),
                    key=f"reb_date_{period_num}"
                )
                rebalance_date = pd.to_datetime(rebalance_date)
        
        # Stock Tickers
        default_tickers = ["AAPL, NVDA, MSFT", "AAPL, NVDA, INTC", "AAPL, GOOGL, TSLA", "AAPL, AMZN, META"]
        ticker_string = st.text_input(
            f"Stock Tickers (Period {period_num})", 
            default_tickers[min(period_idx, 3)],
            key=f"tickers_{period_num}",
            help="Comma-separated ticker symbols"
        )
        tickers = [x.strip().upper() for x in ticker_string.split(',') if x.strip()]
        
        # Weights
        weights = []
        if tickers:
            st.caption(f"Allocation for Period {period_num}")
            cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                weight = cols[i % len(tickers)].number_input(
                    f"{ticker} %", 
                    0, 100, 
                    int(100/len(tickers)), 
                    key=f"weight_{period_num}_{ticker}"
                )
                weights.append(weight)
        
        # Normalize weights
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
        else:
            weights = [1.0/len(tickers)]*len(tickers) if tickers else []
        
        # Store period data
        periods_data.append({
            'period_num': period_num,
            'rebalance_date': rebalance_date,
            'tickers': tickers,
            'weights': weights
        })
    
    run_button = st.form_submit_button("🚀 Run Multi-Period Analysis", use_container_width=True)

# --- ANALYSIS ENGINE ---
if run_button:
    try:
        with st.spinner("Analyzing multi-period portfolio..."):
            
            # Validate dates
            all_dates = [global_start] + [p['rebalance_date'] for p in periods_data if p['rebalance_date']] + [global_end]
            for i in range(len(all_dates) - 1):
                if all_dates[i] >= all_dates[i+1]:
                    st.error(f"⚠️ Dates must be in chronological order! Check Period {i+1} date.")
                    st.stop()
            
            # 1. Calculate Benchmark (full period)
            bench_data = get_stock_data([benchmark_ticker], global_start, global_end)
            if bench_data.empty:
                st.error("❌ No benchmark data found!")
                st.stop()
            if isinstance(bench_data, pd.DataFrame): 
                bench_data = bench_data.iloc[:, 0]
            bench_returns = bench_data.pct_change().dropna()
            bench_cum = (1 + bench_returns).cumprod()
            
            # 2. Calculate Multi-Period Portfolio
            all_cumulative = []
            all_daily_returns = []
            rebalance_dates = []
            current_value = 1.0
            
            for i, period in enumerate(periods_data):
                # Determine start and end for this period
                if i == 0:
                    period_start = global_start
                else:
                    period_start = period['rebalance_date']
                    rebalance_dates.append(period_start)
                
                if i == len(periods_data) - 1:
                    period_end = global_end
                else:
                    period_end = periods_data[i + 1]['rebalance_date']
                
                # Calculate this segment
                cum_segment, daily_segment = calculate_segment_performance(
                    period['tickers'], 
                    period['weights'], 
                    period_start, 
                    period_end,
                    initial_value=current_value
                )
                
                if cum_segment is None or cum_segment.empty:
                    st.error(f"❌ No data for Period {period['period_num']}")
                    st.stop()
                
                # Update current value for next period
                current_value = cum_segment.iloc[-1]
                
                # Store results
                all_cumulative.append(cum_segment)
                all_daily_returns.append(daily_segment)
            
            # 3. Stitch all periods together
            portfolio_cum = pd.concat(all_cumulative)
            portfolio_cum = portfolio_cum[~portfolio_cum.index.duplicated(keep='last')]
            portfolio_daily = pd.concat(all_daily_returns)
            
            # 4. Align with benchmark
            common_idx = portfolio_cum.index.intersection(bench_cum.index)
            portfolio_cum = portfolio_cum.loc[common_idx]
            bench_cum = bench_cum.loc[common_idx]
            
            # 5. Calculate Metrics
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
            st.success(f"✅ Analysis Complete! ({st.session_state.num_periods} periods analyzed)")
            
            # Metrics Row 1
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Return", f"{total_return:.2%}", 
                       delta=f"{(total_return - bench_total_return):.2%} vs Benchmark")
            col2.metric("Portfolio CAGR", f"{portfolio_cagr:.2%}")
            col3.metric("Benchmark Return", f"{bench_total_return:.2%}")
            col4.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
            
            # Metrics Row 2
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
            col6.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}")
            col7.metric("Benchmark Volatility", f"{bench_vol:.2%}")
            col8.metric("Benchmark Sharpe", f"{bench_sharpe:.2f}")
            
            # Metrics Row 3
            col9, col10 = st.columns(2)
            col9.metric("Portfolio Max Drawdown", f"{portfolio_drawdown:.2%}")
            col10.metric("Benchmark Max Drawdown", f"{bench_drawdown:.2%}")
            
            # Chart
            st.subheader("📊 Growth Chart ($1 Investment)")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(portfolio_cum.index, portfolio_cum, label='Your Portfolio', linewidth=2.5, color='#1f77b4')
            ax.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark_ticker})', 
                   linestyle='--', linewidth=2, color='#ff7f0e')
            
            # Add vertical lines for rebalancing dates
            colors = ['purple', 'green', 'red', 'orange', 'brown']
            for idx, reb_date in enumerate(rebalance_dates):
                color = colors[idx % len(colors)]
                ax.axvline(reb_date, color=color, linestyle=':', alpha=0.7, linewidth=2)
                ax.text(reb_date, portfolio_cum.max() * 0.95, 
                       f'  Period {idx+2}', color=color, rotation=90, fontsize=9)
            
            ax.axhline(1.0, color='red', linestyle=':', alpha=0.3)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title(f'Multi-Period Portfolio Performance: {global_start.date()} to {global_end.date()}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed Statistics
            with st.expander("📈 See Detailed Statistics"):
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("**Portfolio Statistics**")
                    st.write(f"• Total Return: {total_return:.2%}")
                    st.write(f"• CAGR (Annualized): {portfolio_cagr:.2%}")
                    st.write(f"• Annualized Volatility: {portfolio_vol:.2%}")
                    st.write(f"• Sharpe Ratio: {portfolio_sharpe:.2f}")
                    st.write(f"• Max Drawdown: {portfolio_drawdown:.2%}")
                    st.write(f"• Best Day: {portfolio_daily.max():.2%}")
                    st.write(f"• Worst Day: {portfolio_daily.min():.2%}")
                    st.write(f"• Time Period: {years:.2f} years")
                    st.write(f"• Number of Rebalances: {len(rebalance_dates)}")
                
                with stats_col2:
                    st.markdown("**Benchmark Statistics**")
                    st.write(f"• Total Return: {bench_total_return:.2%}")
                    st.write(f"• CAGR (Annualized): {bench_cagr:.2%}")
                    st.write(f"• Annualized Volatility: {bench_vol:.2%}")
                    st.write(f"• Sharpe Ratio: {bench_sharpe:.2f}")
                    st.write(f"• Max Drawdown: {bench_drawdown:.2%}")
                    st.write(f"• Best Day: {bench_returns.max():.2%}")
                    st.write(f"• Worst Day: {bench_returns.min():.2%}")
                    st.write(f"• Time Period: {years:.2f} years")
            
            # Period Summary Table
            with st.expander("📋 Period Summary"):
                summary_data = []
                for i, period in enumerate(periods_data):
                    if i == 0:
                        start = global_start.date()
                    else:
                        start = period['rebalance_date'].date()
                    
                    if i == len(periods_data) - 1:
                        end = global_end.date()
                    else:
                        end = periods_data[i + 1]['rebalance_date'].date()
                    
                    summary_data.append({
                        'Period': period['period_num'],
                        'Start Date': start,
                        'End Date': end,
                        'Stocks': ', '.join(period['tickers']),
                        'Allocation': ', '.join([f"{t}:{w:.1%}" for t, w in zip(period['tickers'], period['weights'])])
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Built with ❤️ using Streamlit & yfinance | Multi-Period Portfolio Analysis
</div>
""", unsafe_allow_html=True)
