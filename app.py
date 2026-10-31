import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Fast Intraday ORB Backtester", layout="wide")

st.title("Opening Range Breakout (ORB) Strategy Backtester")
st.markdown("""
A fast intraday backtester using **2-minute intraday data**, configurable commissions, 
**per-ticker compounding equity**, and flexible stop loss rules (Fixed % vs. Opening Range Boundary).
""")

# Sidebar Controls
st.sidebar.header("1. Tickers & Capital")
tickers_input = st.sidebar.text_input(
    "Ticker Symbols (comma-separated)", 
    value="SPY, QQQ, AAPL, MSFT, NVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

total_starting_capital = st.sidebar.number_input("Total Portfolio Starting Capital ($)", value=20000.0, step=1000.0)
num_positions = st.sidebar.number_input("Number of Active Positions / Slots", value=len(tickers), min_value=1, step=1)
initial_capital_per_ticker = total_starting_capital / max(num_positions, 1)

commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.25, format="%.2f")

# Strategy Parameters
st.sidebar.header("2. Strategy Parameters")
or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=2, max_value=60, value=30, step=2)
exit_mode = st.sidebar.selectbox("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])

# Stop Loss Rule Configuration
stop_loss_type = st.sidebar.selectbox(
    "Stop Loss Rule", 
    options=["Fixed Percentage (%)", "Opening Range Boundary (OR Low/High ± 1 tick)"]
)

if stop_loss_type == "Fixed Percentage (%)":
    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.25, max_value=3.00, value=0.50, step=0.25)
else:
    stop_loss_pct = 0.0

if exit_mode == "Profit Target (%)":
    profit_target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.25, max_value=3.00, value=1.00, step=0.25)
    holding_bars = 0
else:
    profit_target_pct = 0.0
    holding_bars = st.sidebar.slider("Holding Bars (2-min intervals)", min_value=5, max_value=120, value=10, step=5)

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

# Load Data
with st.spinner("Fetching intraday data from Yahoo Finance..."):
    all_data = load_all_data(tickers)

if not all_data:
    st.error("No valid data retrieved for any of the specified tickers. Please verify the symbols.")
else:
    raw_trades = []
    num_or_bars = or_minutes // 2

    # Run fast simulation per ticker
    for ticker, data in all_data.items():
        grouped_days = data.groupby(data.index.date)
        
        for date, group in grouped_days:
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
                        
                        # Set Stop Price based on selected rule
                        if stop_loss_type == "Fixed Percentage (%)":
                            stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                        else:
                            stop_price = or_low - 0.01  # Bottom of opening range minus 1 tick ($0.01)
                            
                        if exit_mode == "Profit Target (%)":
                            target_price = entry_price * (1 + profit_target_pct / 100.0)
                        bars_held = 0
                        
                    elif close < or_low:
                        in_position = True
                        direction = 'SHORT'
                        entry_price = close
                        entry_time = bar_time
                        
                        # Set Stop Price based on selected rule
                        if stop_loss_type == "Fixed Percentage (%)":
                            stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                        else:
                            stop_price = or_high + 0.01  # Top of opening range plus 1 tick ($0.01)
                            
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

    if not raw_trades:
        st.warning("No trades were triggered using the current parameters. Try adjusting the range window or thresholds.")
    else:
        # Chronological sorting for equity compounding
        trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
        ticker_equity = {t: initial_capital_per_ticker for t in tickers}
        
        simulated_trades = []
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
            
            simulated_trades.append({
                'Ticker': ticker,
                'Date': row['Date'],
                'Entry Time': row['Entry Time'],
                'Exit Time': row['Exit Time'],
                'Direction': row['Direction'],
                'Shares': shares,
                'Entry Price': row['Entry Price'],
                'Exit Price': row['Exit Price'],
                'Exit Reason': row['Exit Reason'],
                'Net PnL ($)': round(net_pnl, 2),
                'Ending Balance ($)': round(ticker_equity[ticker], 2)
            })
            
        final_trades_df = pd.DataFrame(simulated_trades).sort_values(by='Exit Time').reset_index(drop=True)
        
        # Overall Performance Summary
        st.subheader("Overall Performance Summary")
        total_net_pnl = final_trades_df['Net PnL ($)'].sum()
        win_rate = (final_trades_df['Net PnL ($)'] > 0).mean() * 100
        total_trades = len(final_trades_df)
        winning_trades = len(final_trades_df[final_trades_df['Net PnL ($)'] > 0])
        losing_trades = len(final_trades_df[final_trades_df['Net PnL ($)'] <= 0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Net PnL", f"${total_net_pnl:,.2f}")
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total Trades", total_trades)
        col4.metric("Wins / Losses", f"{winning_trades} / {losing_trades}")
        
        # Long vs Short Breakdown Section
        st.subheader("Long vs. Short Breakdown")
        long_trades = final_trades_df[final_trades_df['Direction'] == 'LONG']
        short_trades = final_trades_df[final_trades_df['Direction'] == 'SHORT']
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("#### 🟢 Long Trades")
            if not long_trades.empty:
                l_pnl = long_trades['Net PnL ($)'].sum()
                l_win_rate = (long_trades['Net PnL ($)'] > 0).mean() * 100
                l_count = len(long_trades)
                l_wins = len(long_trades[long_trades['Net PnL ($)'] > 0])
                l_losses = len(long_trades[long_trades['Net PnL ($)'] <= 0])
                st.metric("Long Net PnL", f"${l_pnl:,.2f}")
                st.metric("Long Win Rate", f"{l_win_rate:.1f}%", f"{l_count} trades ({l_wins}W / {l_losses}L)")
            else:
                st.info("No long trades taken.")
                
        with l_col2:
            st.markdown("#### 🔴 Short Trades")
            if not short_trades.empty:
                s_pnl = short_trades['Net PnL ($)'].sum()
                s_win_rate = (short_trades['Net PnL ($)'] > 0).mean() * 100
                s_count = len(short_trades)
                s_wins = len(short_trades[short_trades['Net PnL ($)'] > 0])
                s_losses = len(short_trades[short_trades['Net PnL ($)'] <= 0])
                st.metric("Short Net PnL", f"${s_pnl:,.2f}")
                st.metric("Short Win Rate", f"{s_win_rate:.1f}%", f"{s_count} trades ({s_wins}W / {s_losses}L)")
            else:
                st.info("No short trades taken.")

        # Ticker Summary Table
        st.subheader("Performance by Ticker")
        ticker_summary = final_trades_df.groupby('Ticker').agg(
            Total_Trades=('Net PnL ($)', 'count'),
            Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean() * 100:.1f}%"),
            Net_PnL=('Net PnL ($)', 'sum'),
            Ending_Balance=('Ending Balance ($)', 'last')
        ).reset_index()
        ticker_summary['Net_PnL'] = ticker_summary['Net_PnL'].apply(lambda x: f"${x:,.2f}")
        ticker_summary['Ending_Balance'] = ticker_summary['Ending_Balance'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(ticker_summary, use_container_width=True)

        # Equity Curve
        final_trades_df['Cumulative PnL'] = final_trades_df['Net PnL ($)'].cumsum()
        st.subheader("Portfolio Equity Curve")
        st.line_chart(final_trades_df.set_index('Exit Time')['Cumulative PnL'])
        
        # Master Trade Log
        st.subheader("Master Trade Log")
        st.dataframe(final_trades_df, use_container_width=True)
