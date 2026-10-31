import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Multi-Stock Intraday Parameter Optimizer & Heatmap", layout="wide")

st.title("Multi-Stock Intraday Strategy Heatmap & Parameter Optimizer")
st.markdown("""
Run batch parameter sweeps across multiple stocks using **2-minute intraday data**, configurable commissions, 
per-ticker compounding equity, and flexible exit modes (**Profit Targets** or **Bar Holds**). 
Results are displayed in an interactive performance matrix.
""")

# Sidebar Controls
st.sidebar.header("1. Strategy & Scope")
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

# Parameter Sweep Ranges & Exit Mode Selection
st.sidebar.header("3. Parameter Sweep Ranges")
exit_mode = st.sidebar.selectbox("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])

or_minutes_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
stop_loss_list = [0.25, 0.50, 0.75, 1.00]

if exit_mode == "Profit Target (%)":
    sweep_param_list = [0.5, 1.0, 1.5, 2.0]  # Profit targets in %
    param_label = "Profit Target (%)"
else:
    sweep_param_list = [5, 10, 15, 20]     # Number of 2-minute bars to hold
    param_label = "Holding Bars"

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

def run_backtest_simulation(all_data, p_or_mins, p_stop, p_val):
    raw_trades = []
    
    for ticker, data in all_data.items():
        grouped_days = data.groupby(data.index.date)
        
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
                            target_price = entry_price * (1 + p_val / 100.0)
                        bars_held = 0
                    elif close < or_low:
                        in_position = True
                        direction = 'SHORT'
                        entry_price = close
                        entry_time = bar_time
                        stop_price = entry_price * (1 + p_stop / 100.0)
                        if exit_mode == "Profit Target (%)":
                            target_price = entry_price * (1 - p_val / 100.0)
                        bars_held = 0
                else:
                    bars_held += 1
                    exit_price = None
                    
                    if direction == 'LONG':
                        if low <= stop_price:
                            exit_price = stop_price
                        elif exit_mode == "Profit Target (%)" and high >= target_price:
                            exit_price = target_price
                        elif exit_mode == "Hold for X Bars" and bars_held >= p_val:
                            exit_price = close
                        elif i == len(trading_slice) - 1:
                            exit_price = close
                    elif direction == 'SHORT':
                        if high >= stop_price:
                            exit_price = stop_price
                        elif exit_mode == "Profit Target (%)" and low <= target_price:
                            exit_price = target_price
                        elif exit_mode == "Hold for X Bars" and bars_held >= p_val:
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

all_data = load_all_data(tickers)

if not all_data:
    st.error("No valid data retrieved for any of the specified tickers. Please verify the symbols.")
else:
    if st.button("Run Parameter Sweep Matrix", type="primary"):
        with st.spinner(f"Running parameter combinations across {exit_mode.lower()} values, stop losses, and durations..."):
            sweep_results = []
            
            for or_m in or_minutes_list:
                for sl in stop_loss_list:
                    for val in sweep_param_list:
                        pnl, wr, count = run_backtest_simulation(all_data, or_m, sl, val)
                        sweep_results.append({
                            'OR_Minutes': or_m,
                            'Stop_Loss': sl,
                            param_label: val,
                            'Net_PnL': round(pnl, 2),
                            'Win_Rate': round(wr, 1),
                            'Trades': count
                        })
            
            sweep_df = pd.DataFrame(sweep_results)
            
            st.subheader("Complete Sweep Results Table")
            st.dataframe(sweep_df.sort_values(by="Net_PnL", ascending=False), use_container_width=True)
            
            # Interactive Matrix Slice for Heatmap/Table View
            st.subheader("Interactive Heatmap Matrix Slice")
            selected_param = st.selectbox(f"Select {param_label} to view matrix slice:", options=sweep_param_list)
            
            filtered_df = sweep_df[sweep_df[param_label] == selected_param]
            matrix_df = filtered_df.pivot(index="OR_Minutes", columns="Stop_Loss", values="Net_PnL")
            
            st.markdown(f"**Net PnL Heatmap Matrix for {param_label} = {selected_param}** (Rows: OR Duration, Columns: Stop Loss)")
            st.dataframe(
                matrix_df.style.background_gradient(cmap="RdYlGn", axis=None), 
                use_container_width=True
            )
