import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Multi-Stock Intraday Parameter Optimizer & Heatmap", layout="wide")

st.title("Multi-Stock Intraday Strategy Heatmap & Parameter Optimizer")
st.markdown("""
Run batch parameter sweeps across multiple stocks using **2-minute intraday data**, configurable commissions, 
and per-ticker compounding equity. Results are aggregated into an interactive **Heatmap**.
""")

# Sidebar Controls
st.sidebar.header("1. Strategy & Scope")
strategy_type = st.sidebar.selectbox(
    "Choose Strategy", 
    options=["Opening Range Breakout (ORB)", "Open-Percent Threshold Strategy"]
)

tickers_input = st.sidebar.text_input(
    "Ticker Symbols (comma-separated)", 
    value="SPY, QQQ, AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.header("2. Capital & Execution")
total_starting_capital = st.sidebar.number_input("Total Portfolio Starting Capital ($)", value=20000.0, step=1000.0)
num_positions = st.sidebar.number_input("Number of Active Positions / Slots", value=len(tickers), min_value=1, step=1)
initial_capital_per_ticker = total_starting_capital / max(num_positions, 1)

# Editable commission control
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.25, format="%.2f")

# Parameter Sweep Ranges based on strategy
st.sidebar.header("3. Parameter Sweep Ranges")
if strategy_type == "Opening Range Breakout (ORB)":
    exit_mode = st.sidebar.selectbox("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])
    
    # Sweep ranges as requested
    or_minutes_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    stop_loss_list = [0.25, 0.50, 0.75, 1.00]
    
    if exit_mode == "Profit Target (%)":
        target_list = [0.5, 1.0, 1.5, 2.0]
        holding_bars_list = [0]
    else:
        target_list = [0.0]
        holding_bars_list = [5, 10, 15, 20]
else:
    trade_direction = st.sidebar.selectbox("Trade Direction", options=["Both (Long & Short)", "Long Only", "Short Only"])
    entry_threshold_list = [0.1, 0.25, 0.5]
    stop_loss_list = [0.25, 0.50, 0.75, 1.00]
    target_list = [0.5, 1.0, 1.5, 2.0]
    holding_bars_list = [0]

@st.cache_data(ttl=3600)
def load_all_data(symbol_list):
    data_dict = {}
    for symbol in symbol_list:
        df = yf.download(symbol, period="60d", interval="2m", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            df = df.between_time('09:30', '16:00')
            data_dict[symbol] = df
    return data_dict

# Helper function to run a backtest given specific parameters
def run_backtest_simulation(all_data, p_or_mins, p_stop, p_target, p_bars):
    raw_trades = []
    
    for ticker, data in all_data.items():
        grouped_days = data.groupby(data.index.date)
        
        if strategy_type == "Opening Range Breakout (ORB)":
            for date, group in grouped_days:
                num_or_bars = p_or_mins // 2
                if len(group) < num_or_bars + 1:
                    continue
                
                or_slice = group.iloc[:num_or_bars]
                or_high = or_slice['High'].max()
                or_low = or_slice['Low'].min()
                
                trading_slice = group.iloc[num_or_bars:]
                in_position = False
                entry_price = 0.0
                entry_time = None
                direction = None 
                target_price = 0.0
                stop_price = 0.0
                bars_held = 0
                
                for i in range(len(trading_slice)):
                    bar = trading_slice.iloc[i]
                    bar_time = trading_slice.index[i]
                    close = bar['Close']
                    high = bar['High']
                    low = bar['Low']
                    
                    if not in_position:
                        if close > or_high:
                            in_position = True
                            direction = 'LONG'
                            entry_price = close
                            entry_time = bar_time
                            stop_price = entry_price * (1 - p_stop / 100.0)
                            if exit_mode == "Profit Target (%)":
                                target_price = entry_price * (1 + p_target / 100.0)
                            bars_held = 0
                        elif close < or_low:
                            in_position = True
                            direction = 'SHORT'
                            entry_price = close
                            entry_time = bar_time
                            stop_price = entry_price * (1 + p_stop / 100.0)
                            if exit_mode == "Profit Target (%)":
                                target_price = entry_price * (1 - p_target / 100.0)
                            bars_held = 0
                    else:
                        bars_held += 1
                        exit_price = None
                        
                        if direction == 'LONG':
                            if low <= stop_price:
                                exit_price = stop_price
                            elif exit_mode == "Profit Target (%)" and high >= target_price:
                                exit_price = target_price
                            elif exit_mode == "Hold for X Bars" and bars_held >= p_bars:
                                exit_price = close
                            elif i == len(trading_slice) - 1:
                                exit_price = close
                        elif direction == 'SHORT':
                            if high >= stop_price:
                                exit_price = stop_price
                            elif exit_mode == "Profit Target (%)" and low <= target_price:
                                exit_price = target_price
                            elif exit_mode == "Hold for X Bars" and bars_held >= p_bars:
                                exit_price = close
                            elif i == len(trading_slice) - 1:
                                exit_price = close
                        
                        if exit_price is not None:
                            raw_trades.append({
                                'Ticker': ticker,
                                'Date': date,
                                'Entry Time': entry_time,
                                'Exit Time': bar_time,
                                'Direction': direction,
                                'Entry Price': entry_price,
                                'Exit Price': exit_price
                            })
                            in_position = False
                            break
        else:
            # Open-percent threshold logic sweep stub if needed, or keep core ORB sweep active
            pass

    if not raw_trades:
        return 0.0, 0.0, 0
        
    trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
    ticker_equity = {t: initial_capital_per_ticker for t in tickers}
    total_net_pnl = 0.0
    winning_trades = 0
    
    for _, row in trades_df.iterrows():
        ticker = row['Ticker']
        if ticker not in ticker_equity:
            ticker_equity[ticker] = initial_capital_per_ticker
            
        current_capital = max(ticker_equity[ticker], 100.0)
        shares = int(round(current_capital / row['Entry Price']))
        if shares < 1:
            shares = 1
            
        if row['Direction'] == 'LONG':
            gross_pnl = (row['Exit Price'] - row['Entry Price']) * shares
        else:
            gross_pnl = (row['Entry Price'] - row['Exit Price']) * shares
            
        net_pnl = gross_pnl - commission_per_trade
        ticker_equity[ticker] += net_pnl
        total_net_pnl += net_pnl
        if net_pnl > 0:
            winning_trades += 1
            
    win_rate = (winning_trades / len(trades_df)) * 100
    return total_net_pnl, win_rate, len(trades_df)

# Load Data for all tickers
all_data = load_all_data(tickers)

if not all_data:
    st.error("No valid data retrieved for any of the specified tickers. Please verify the symbols.")
else:
    if st.button("Run Parameter Sweep & Generate Heatmap", type="primary"):
        with st.spinner("Running parameter combinations..."):
            sweep_results = []
            
            if strategy_type == "Opening Range Breakout (ORB)":
                for or_m in or_minutes_list:
                    for sl in stop_loss_list:
                        target_val = target_list[0] if exit_mode == "Profit Target (%)" else 0.0
                        bars_val = holding_bars_list[0] if exit_mode == "Hold for X Bars" else 0
                        
                        pnl, wr, count = run_backtest_simulation(all_data, or_m, sl, target_val, bars_val)
                        sweep_results.append({
                            'OR_Minutes': or_m,
                            'Stop_Loss': sl,
                            'Net_PnL': pnl,
                            'Win_Rate': wr,
                            'Trades': count
                        })
                
                sweep_df = pd.DataFrame(sweep_results)
                
                # Pivot table for Heatmap (X-axis: Stop Loss, Y-axis: OR Minutes)
                heatmap_data = sweep_df.pivot(index="OR_Minutes", columns="Stop_Loss", values="Net_PnL")
                
                st.subheader("Performance Heatmap (Net PnL by OR Duration vs. Stop Loss)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={'label': 'Net PnL ($)'})
                ax.set_title("Total Portfolio Net PnL Heatmap")
                ax.set_xlabel("Stop Loss (%)")
                ax.set_ylabel("Opening Range Duration (Minutes)")
                st.pyplot(fig)
                
                st.subheader("Sweep Results Data Table")
                st.dataframe(sweep_df, use_container_width=True)
            else:
                st.info("Heatmap matrix view is optimized for ORB duration vs Stop Loss sweeps.")
