import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Opening Range Breakout Backtester", layout="wide")

st.title("Opening Range Breakout (ORB) System Backtester")
st.markdown("""
This application runs an **Opening Range Breakout (ORB)** strategy using **2-minute intraday data** 
(the maximum lookback window for 2-minute data on Yahoo Finance is **60 days**). 
It models profit targets, stop losses, and **$1 commissions per trade execution**.
""")

# Sidebar Controls for Strategy Parameters
st.sidebar.header("Strategy Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_val=2, max_val=60, value=30, step=2)
profit_target_pct = st.sidebar.slider("Profit Target (%)", min_val=0.1, max_val=5.0, value=1.0, step=0.1)
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_val=0.1, max_val=5.0, value=0.5, step=0.1)
position_size_shares = st.sidebar.number_input("Shares / Contracts per Trade", value=100, step=10)
commission_per_order = 1.0  # $1 commission per order leg (Entry & Exit)

@st.cache_data(ttl=3600)
def load_intraday_data(symbol):
    # Fetching max allowed historical 2-minute data (60 days)
    df = yf.download(symbol, period="60d", interval="2m", progress=False)
    if df.empty:
        return df
    # Clean up multi-index columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Load Data
data = load_intraday_data(ticker)

if data.empty:
    st.error(f"No data retrieved for ticker '{ticker}'. Please verify the symbol or try again later.")
else:
    # Ensure proper timezone localization (Eastern Time for US markets)
    if data.index.tz is not None:
        data.index = data.index.tz_convert('America/New_York')
    else:
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
        
    # Filter for regular trading hours (9:30 AM - 4:00 PM EST)
    data = data.between_time('09:30', '16:00')
    
    trades = []
    grouped_days = data.groupby(data.index.date)
    
    # Backtest Loop over each trading day
    for date, group in grouped_days:
        num_or_bars = or_minutes // 2
        if len(group) < num_or_bars + 1:
            continue
            
        # 1. Define Opening Range (OR) High and Low based on the initial N minutes
        or_slice = group.iloc[:num_or_bars]
        or_high = or_slice['High'].max()
        or_low = or_slice['Low'].min()
        
        # 2. Scanning subsequent bars for breakout triggers
        trading_slice = group.iloc[num_or_bars:]
        
        in_position = False
        entry_price = 0.0
        entry_time = None
        direction = None 
        target_price = 0.0
        stop_price = 0.0
        
        for i in range(len(trading_slice)):
            bar = trading_slice.iloc[i]
            bar_time = trading_slice.index[i]
            close = bar['Close']
            high = bar['High']
            low = bar['Low']
            
            if not in_position:
                # Long entry: First close above Opening Range High
                if close > or_high:
                    in_position = True
                    direction = 'LONG'
                    entry_price = close
                    entry_time = bar_time
                    target_price = entry_price * (1 + profit_target_pct / 100.0)
                    stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                    
                # Short entry: First close below Opening Range Low
                elif close < or_low:
                    in_position = True
                    direction = 'SHORT'
                    entry_price = close
                    entry_time = bar_time
                    target_price = entry_price * (1 - profit_target_pct / 100.0)
                    stop_price = entry_price * (1 + stop_loss_pct / 100.0)
            else:
                # Active Position Management
                exit_price = None
                exit_reason = None
                
                if direction == 'LONG':
                    if high >= target_price:
                        exit_price = target_price
                        exit_reason = 'Profit Target'
                    elif low <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'Stop Loss'
                    elif i == len(trading_slice) - 1: # Market Close Exit (EOD)
                        exit_price = close
                        exit_reason = 'EOD Close'
                elif direction == 'SHORT':
                    if low <= target_price:
                        exit_price = target_price
                        exit_reason = 'Profit Target'
                    elif high >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'Stop Loss'
                    elif i == len(trading_slice) - 1: # Market Close Exit (EOD)
                        exit_price = close
                        exit_reason = 'EOD Close'
                
                if exit_price is not None:
                    # Calculate PnL ($1 commission per order leg: Entry + Exit = $2 total)
                    if direction == 'LONG':
                        gross_pnl = (exit_price - entry_price) * position_size_shares
                    else:
                        gross_pnl = (entry_price - exit_price) * position_size_shares
                        
                    total_commission = commission_per_order * 2
                    net_pnl = gross_pnl - total_commission
                    
                    trades.append({
                        'Date': date,
                        'Entry Time': entry_time,
                        'Exit Time': bar_time,
                        'Direction': direction,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Exit Reason': exit_reason,
                        'Gross PnL ($)': round(gross_pnl, 2),
                        'Net PnL ($)': round(net_pnl, 2)
                    })
                    in_position = False
                    break  # Take max 1 trade per day for this ORB system logic

    trades_df = pd.DataFrame(trades)
    
    if trades_df.empty:
        st.warning("No trades were triggered using the current parameters. Try adjusting the range window or target thresholds.")
    else:
        # Performance Metrics Display
        st.subheader("Performance Summary")
        total_net_pnl = trades_df['Net PnL ($)'].sum()
        win_rate = (trades_df['Net PnL ($)'] > 0).mean() * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['Net PnL ($)'] > 0])
        losing_trades = len(trades_df[trades_df['Net PnL ($)'] <= 0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Net PnL", f"${total_net_pnl:,.2f}")
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total Trades", total_trades)
        col4.metric("Wins / Losses", f"{winning_trades} / {losing_trades}")
        
        # Equity Curve Visualization
        trades_df['Cumulative Net PnL'] = trades_df['Net PnL ($)'].cumsum()
        st.subheader("Equity Curve")
        st.line_chart(trades_df.set_index('Exit Time')['Cumulative Net PnL'])
        
        # Detailed Trade Log
        st.subheader("Detailed Trade Log")
        st.dataframe(trades_df, use_container_width=True)
