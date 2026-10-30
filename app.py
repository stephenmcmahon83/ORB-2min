import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Opening Range Breakout Backtester", layout="wide")

st.title("Opening Range Breakout (ORB) System Backtester")
st.markdown("""
This application runs an **Opening Range Breakout (ORB)** strategy using **2-minute intraday data** 
(maximum lookback on Yahoo Finance is **60 days**). It models stop losses, fixed commissions ($1 per execution), 
and allows toggling between a **Profit Target (%)** or **Holding for a fixed number of bars**.
""")

# Sidebar Controls for Strategy Parameters
st.sidebar.header("Strategy Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=2, max_value=60, value=30, step=2)

# Exit Mode Selection
exit_mode = st.sidebar.radio("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])

stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

if exit_mode == "Profit Target (%)":
    profit_target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    holding_bars = 0
else:
    profit_target_pct = 0.0
    holding_bars = st.sidebar.slider("Hold Duration (Number of 2-min Bars)", min_value=1, max_value=60, value=10, step=1)

position_size_shares = st.sidebar.number_input("Shares / Contracts per Trade", value=100, step=10)
commission_per_order = 1.0  # $1 commission per order leg (Entry & Exit)

@st.cache_data(ttl=3600)
def load_intraday_data(symbol):
    df = yf.download(symbol, period="60d", interval="2m", progress=False)
    if df.empty:
        return df
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
        bars_held = 0
        
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
                    stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                    if exit_mode == "Profit Target (%)":
                        target_price = entry_price * (1 + profit_target_pct / 100.0)
                    bars_held = 0
                    
                # Short entry: First close below Opening Range Low
                elif close < or_low:
                    in_position = True
                    direction = 'SHORT'
                    entry_price = close
                    entry_time = bar_time
                    stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                    if exit_mode == "Profit Target (%)":
                        target_price = entry_price * (1 - profit_target_pct / 100.0)
                    bars_held = 0
            else:
                bars_held += 1
                exit_price = None
                exit_reason = None
                
                if direction == 'LONG':
                    # Check Stop Loss first (conservative intraday check)
                    if low <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'Stop Loss'
                    elif exit_mode == "Profit Target (%)" and high >= target_price:
                        exit_price = target_price
                        exit_reason = 'Profit Target'
                    elif exit_mode == "Hold for X Bars" and bars_held >= holding_bars:
                        exit_price = close
                        exit_reason = f'Bar Count ({holding_bars})'
                    elif i == len(trading_slice) - 1: # Market Close Exit (EOD)
                        exit_price = close
                        exit_reason = 'EOD Close'
                        
                elif direction == 'SHORT':
                    if high >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'Stop Loss'
                    elif exit_mode == "Profit Target (%)" and low <= target_price:
                        exit_price = target_price
                        exit_reason = 'Profit Target'
                    elif exit_mode == "Hold for X Bars" and bars_held >= holding_bars:
                        exit_price = close
                        exit_reason = f'Bar Count ({holding_bars})'
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
        # Overall Performance Metrics Display
        st.subheader("Overall Performance Summary")
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
        
        # Long vs Short Breakdown Section
        st.subheader("Long vs. Short Breakdown")
        long_trades = trades_df[trades_df['Direction'] == 'LONG']
        short_trades = trades_df[trades_df['Direction'] == 'SHORT']
        
        l_col1, l_col2 = st.columns(2)
        
        with l_col1:
            st.markdown("#### 🟢 Long Trades")
            if not long_trades.empty:
                l_pnl = long_trades['Net PnL ($)'].sum()
                l_win_rate = (long_trades['Net PnL ($)'] > 0).mean() * 100
                l_count = len(long_trades)
                st.metric("Long Net PnL", f"${l_pnl:,.2f}")
                st.metric("Long Win Rate", f"{l_win_rate:.1f}%", f"{l_count} trades")
            else:
                st.info("No long trades taken.")
                
        with l_col2:
            st.markdown("#### 🔴 Short Trades")
            if not short_trades.empty:
                s_pnl = short_trades['Net PnL ($)'].sum()
                s_win_rate = (short_trades['Net PnL ($)'] > 0).mean() * 100
                s_count = len(short_trades)
                st.metric("Short Net PnL", f"${s_pnl:,.2f}")
                st.metric("Short Win Rate", f"{s_win_rate:.1f}%", f"{s_count} trades")
            else:
                st.info("No short trades taken.")

        # Equity Curve Visualization
        trades_df['Cumulative Net PnL'] = trades_df['Net PnL ($)'].cumsum()
        st.subheader("Equity Curve")
        st.line_chart(trades_df.set_index('Exit Time')['Cumulative Net PnL'])
        
        # Detailed Trade Log
        st.subheader("Detailed Trade Log")
        st.dataframe(trades_df, use_container_width=True)
