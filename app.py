import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Multi-Stock Intraday Strategy Backtester", layout="wide")

st.title("Multi-Stock Intraday Trading System Backtester")
st.markdown("""
Backtest an entire basket of stocks simultaneously using **2-minute intraday data** 
(maximum lookback on Yahoo Finance is **60 days**), incorporating a flat **$1.00 commission per trade**. 
Profits and losses are tracked and reinvested **per ticker independently** using whole shares.
""")

# Sidebar Controls for Strategy Selection
st.sidebar.header("Strategy Selection")
strategy_type = st.sidebar.selectbox(
    "Choose Strategy", 
    options=["Opening Range Breakout (ORB)", "Open-Percent Threshold Strategy"]
)

# Multi-stock ticker input
tickers_input = st.sidebar.text_input(
    "Ticker Symbols (comma-separated)", 
    value="SPY, QQQ, AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Portfolio & Capital Management
st.sidebar.header("Capital & Position Sizing")
total_starting_capital = st.sidebar.number_input("Total Portfolio Starting Capital ($)", value=20000.0, step=1000.0)
num_positions = st.sidebar.number_input("Number of Active Positions / Slots", value=len(tickers), min_value=1, step=1)
initial_capital_per_ticker = total_starting_capital / max(num_positions, 1)
st.sidebar.markdown(f"**Initial Capital per Ticker:** ${initial_capital_per_ticker:,.2f}")

commission_per_trade = 1.0  # Flat $1 commission per trade

# Dynamic Sidebar Parameters Based on Strategy
if strategy_type == "Opening Range Breakout (ORB)":
    st.sidebar.header("ORB Configuration")
    or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=2, max_value=60, value=30, step=2)
    exit_mode = st.sidebar.radio("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])
    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    if exit_mode == "Profit Target (%)":
        profit_target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        holding_bars = 0
    else:
        profit_target_pct = 0.0
        holding_bars = st.sidebar.slider("Hold Duration (Number of 2-min Bars)", min_value=1, max_value=60, value=10, step=1)
else:
    st.sidebar.header("Open-Percent Strategy Configuration")
    trade_direction = st.sidebar.selectbox("Trade Direction", options=["Both (Long & Short)", "Long Only", "Short Only"])
    entry_threshold_pct = st.sidebar.slider("Entry Threshold (% from Open)", min_value=0.05, max_value=2.0, value=0.25, step=0.05)
    target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    stop_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.05, max_value=5.0, value=0.25, step=0.05)

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

# Load Data for all tickers
all_data = load_all_data(tickers)

if not all_data:
    st.error("No valid data retrieved for any of the specified tickers. Please verify the symbols.")
else:
    raw_trades = []

    # Step 1: Generate all raw trades per ticker independently
    for ticker, data in all_data.items():
        grouped_days = data.groupby(data.index.date)
        
        if strategy_type == "Opening Range Breakout (ORB)":
            for date, group in grouped_days:
                num_or_bars = or_minutes // 2
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
                            stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                            if exit_mode == "Profit Target (%)":
                                target_price = entry_price * (1 + profit_target_pct / 100.0)
                            bars_held = 0
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
                            if low <= stop_price:
                                exit_price = stop_price
                                exit_reason = 'Stop Loss'
                            elif exit_mode == "Profit Target (%)" and high >= target_price:
                                exit_price = target_price
                                exit_reason = 'Profit Target'
                            elif exit_mode == "Hold for X Bars" and bars_held >= holding_bars:
                                exit_price = close
                                exit_reason = f'Bar Count ({holding_bars})'
                            elif i == len(trading_slice) - 1:
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
                            elif i == len(trading_slice) - 1:
                                exit_price = close
                                exit_reason = 'EOD Close'
                        
                        if exit_price is not None:
                            raw_trades.append({
                                'Ticker': ticker,
                                'Date': date,
                                'Entry Time': entry_time,
                                'Exit Time': bar_time,
                                'Direction': direction,
                                'Entry Price': entry_price,
                                'Exit Price': exit_price,
                                'Exit Reason': exit_reason
                            })
                            in_position = False
                            break
        else:
            for date, group in grouped_days:
                if len(group) < 2:
                    continue
                
                day_open = group.iloc[0]['Open']
                in_position = False
                entry_price = 0.0
                entry_time = None
                direction = None
                target_price = 0.0
                stop_price = 0.0
                
                for i in range(len(group)):
                    bar = group.iloc[i]
                    bar_time = group.index[i]
                    close = bar['Close']
                    
                    if not in_position:
                        if trade_direction in ["Both (Long & Short)", "Long Only"] and close >= day_open * (1 + entry_threshold_pct / 100.0):
                            in_position = True
                            direction = 'LONG'
                            entry_price = close
                            entry_time = bar_time
                            target_price = entry_price * (1 + target_pct / 100.0)
                            stop_price = entry_price * (1 - stop_pct / 100.0)
                        elif trade_direction in ["Both (Long & Short)", "Short Only"] and close <= day_open * (1 - entry_threshold_pct / 100.0):
                            in_position = True
                            direction = 'SHORT'
                            entry_price = close
                            entry_time = bar_time
                            target_price = entry_price * (1 - target_pct / 100.0)
                            stop_price = entry_price * (1 + stop_pct / 100.0)
                    else:
                        exit_price = None
                        exit_reason = None
                        
                        if direction == 'LONG':
                            if close >= target_price:
                                exit_price = close
                                exit_reason = 'Profit Target (Close)'
                            elif close <= stop_price:
                                exit_price = close
                                exit_reason = 'Stop Loss (Close)'
                            elif i == len(group) - 1:
                                exit_price = close
                                exit_reason = 'EOD Close'
                        elif direction == 'SHORT':
                            if close <= target_price:
                                exit_price = close
                                exit_reason = 'Profit Target (Close)'
                            elif close >= stop_price:
                                exit_price = close
                                exit_reason = 'Stop Loss (Close)'
                            elif i == len(group) - 1:
                                exit_price = close
                                exit_reason = 'EOD Close'
                            
                        if exit_price is not None:
                            raw_trades.append({
                                'Ticker': ticker,
                                'Date': date,
                                'Entry Time': entry_time,
                                'Exit Time': bar_time,
                                'Direction': direction,
                                'Entry Price': entry_price,
                                'Exit Price': exit_price,
                                'Exit Reason': exit_reason
                            })
                            in_position = False
                            break

    if not raw_trades:
        st.warning("No trades were triggered across any of the selected tickers using current parameters.")
    else:
        # Step 2: Sort all trades chronologically by Entry Time
        trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
        
        # Dictionary to track rolling capital balance per ticker independently
        ticker_equity = {t: initial_capital_per_ticker for t in tickers}
        
        simulated_trades = []
        
        for _, row in trades_df.iterrows():
            ticker = row['Ticker']
            
            # Ensure ticker exists in dictionary (in case user added custom symbols)
            if ticker not in ticker_equity:
                ticker_equity[ticker] = initial_capital_per_ticker
                
            current_ticker_capital = max(ticker_equity[ticker], 100.0) # Floor capital at $100 minimum to avoid zero division
            
            # Calculate whole shares based on this ticker's rolling balance
            shares = int(round(current_ticker_capital / row['Entry Price']))
            if shares < 1:
                shares = 1  # Force minimum 1 share if capital is low
                
            # Calculate PnL
            if row['Direction'] == 'LONG':
                gross_pnl = (row['Exit Price'] - row['Entry Price']) * shares
            else:
                gross_pnl = (row['Entry Price'] - row['Exit Price']) * shares
                
            net_pnl = gross_pnl - commission_per_trade
            
            # Reinvest profits/losses into that specific ticker's balance
            ticker_equity[ticker] += net_pnl
            
            simulated_trades.append({
                'Ticker': ticker,
                'Date': row['Date'],
                'Entry Time': row['Entry Time'],
                'Exit Time': row['Exit Time'],
                'Direction': row['Direction'],
                'Ticker Balance Before ($)': round(current_ticker_capital, 2),
                'Shares Traded': shares,
                'Entry Price': row['Entry Price'],
                'Exit Price': row['Exit Price'],
                'Exit Reason': row['Exit Reason'],
                'Gross PnL ($)': round(gross_pnl, 2),
                'Net PnL ($)': round(net_pnl, 2),
                'Ticker Balance After ($)': round(ticker_equity[ticker], 2)
            })
            
        final_trades_df = pd.DataFrame(simulated_trades).sort_values(by='Exit Time').reset_index(drop=True)
        
        # Portfolio Performance Metrics Display
        st.subheader("Portfolio Performance Summary")
        total_net_pnl = final_trades_df['Net PnL ($)'].sum()
        win_rate = (final_trades_df['Net PnL ($)'] > 0).mean() * 100
        total_trades = len(final_trades_df)
        winning_trades = len(final_trades_df[final_trades_df['Net PnL ($)'] > 0])
        losing_trades = len(final_trades_df[final_trades_df['Net PnL ($)'] <= 0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Total Net PnL", f"${total_net_pnl:,.2f}")
        col2.metric("Portfolio Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total Trades Executed", total_trades)
        col4.metric("Wins / Losses", f"${winning_trades} / {losing_trades}".replace('$', ''))
        
        # Breakdown: Long vs Short Portfolio PnL
        st.subheader("Directional Breakdown")
        long_trades = final_trades_df[final_trades_df['Direction'] == 'LONG']
        short_trades = final_trades_df[final_trades_df['Direction'] == 'SHORT']
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("#### 🟢 Portfolio Longs")
            if not long_trades.empty:
                st.metric("Long Net PnL", f"${long_trades['Net PnL ($)'].sum():,.2f}", f"{len(long_trades)} trades")
            else:
                st.info("No long trades.")
        with l_col2:
            st.markdown("#### 🔴 Portfolio Shorts")
            if not short_trades.empty:
                st.metric("Short Net PnL", f"${short_trades['Net PnL ($)'].sum():,.2f}", f"{len(short_trades)} trades")
            else:
                st.info("No short trades.")

        # Breakdown by Ticker (Showing Ending Balance per Ticker)
        st.subheader("Performance & Reinvestment Breakdown by Ticker")
        ticker_summary = final_trades_df.groupby('Ticker').agg(
            Total_Trades=('Net PnL ($)', 'count'),
            Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean() * 100:.1f}%"),
            Net_PnL=('Net PnL ($)', 'sum'),
            Ending_Balance=('Ticker Balance After ($)', 'last')
        ).reset_index()
        ticker_summary['Net_PnL'] = ticker_summary['Net_PnL'].apply(lambda x: f"${x:,.2f}")
        ticker_summary['Ending_Balance'] = ticker_summary['Ending_Balance'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(ticker_summary, use_container_width=True)

        # Portfolio Equity Curve Visualization
        final_trades_df['Cumulative Portfolio PnL'] = final_trades_df['Net PnL ($)'].cumsum()
        st.subheader("Combined Portfolio Equity Curve")
        st.line_chart(final_trades_df.set_index('Exit Time')['Cumulative Portfolio PnL'])
        
        # Combined Detailed Trade Log
        st.subheader("Master Trade Log")
        st.dataframe(final_trades_df, use_container_width=True)
