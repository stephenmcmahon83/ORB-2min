import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="SPY CSV ORB Backtester", layout="wide")

st.title("Opening Range Breakout (ORB) Strategy Backtester")
st.markdown("""
Upload your 1-minute SPY CSV file. This version flexible-starts trading on the **first available bar** of each day, 
bypassing strict 9:30 AM grid requirements so your dataset generates results.
""")

# Sidebar Controls
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload SPY CSV File", type=["csv"])

st.sidebar.header("2. Capital & Execution")
starting_capital = st.sidebar.number_input("Starting Capital ($)", value=20000.0, step=1000.0)
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.25, format="%.2f")

# Strategy Parameters
st.sidebar.header("3. Strategy Parameters")
or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=1, max_value=60, value=15, step=1)
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
    holding_bars = st.sidebar.slider("Holding Bars (1-min intervals)", min_value=1, max_value=240, value=10, step=1)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    with st.expander("🔍 View Raw Uploaded Data Preview"):
        st.write("Detected Columns:", list(df.columns))
        st.dataframe(df.head(), use_container_width=True)

    try:
        datetime_col_name = df.columns[0]
        df[datetime_col_name] = pd.to_datetime(df[datetime_col_name], errors='coerce')
        df = df.dropna(subset=[datetime_col_name])
        df.set_index(datetime_col_name, inplace=True)
        
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(df.columns) >= 5:
            df = df.iloc[:, :5]
            df.columns = expected_cols
        else:
            df.columns = [str(c).capitalize() for c in df.columns[:len(expected_cols)]]
            
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    except Exception as e:
        st.error(f"Error parsing datetime or price columns: {e}")
        st.stop()

    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York')
    else:
        try:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        except Exception:
            pass 
            
    df = df.between_time('09:30', '16:00')
    
    if df.empty:
        st.error("No data remaining after filtering for regular trading hours. Check your timestamp formatting.")
        st.stop()

    raw_trades = []
    num_or_bars = or_minutes  # Number of 1-min bars for the opening range
    
    grouped_days = df.groupby(df.index.date)
    days_backtested = 0
    
    for date, group in grouped_days:
        group = group[~group.index.duplicated(keep='first')]
        
        # Ensure there are enough bars in the day to form an opening range + at least 1 trade bar
        if len(group) < num_or_bars + 1:
            continue
            
        days_backtested += 1
        
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
                    
                    if stop_loss_type == "Fixed Percentage (%)":
                        stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                    else:
                        stop_price = or_low - 0.01
                        
                    if exit_mode == "Profit Target (%)":
                        target_price = entry_price * (1 + profit_target_pct / 100.0)
                    bars_held = 0
                    
                elif close < or_low:
                    in_position = True
                    direction = 'SHORT'
                    entry_price = close
                    entry_time = bar_time
                    
                    if stop_loss_type == "Fixed Percentage (%)":
                        stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                    else:
                        stop_price = or_high + 0.01
                        
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

    st.success(f"Successfully backtested **{days_backtested} trading days** from your uploaded file!")

    if not raw_trades:
        st.warning("No trades were triggered using the current parameters. Try adjusting your Opening Range duration, stop loss, or profit target.")
    else:
        trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
        
        current_equity = starting_capital
        simulated_trades = []
        
        for _, row in trades_df.iterrows():
            capital_to_risk = max(current_equity, 100.0)
            shares = int(round(capital_to_risk / row['Entry Price']))
            if shares < 1:
                shares = 1
                
            if row['Direction'] == 'LONG':
                gross_pnl = (row['Exit Price'] - row['Entry Price']) * shares
            else:
                gross_pnl = (row['Entry Price'] - row['Exit Price']) * shares
                
            net_pnl = gross_pnl - commission_per_trade
            current_equity += net_pnl
            
            simulated_trades.append({
                'Date': row['Date'],
                'Entry Time': row['Entry Time'],
                'Exit Time': row['Exit Time'],
                'Direction': row['Direction'],
                'Shares': shares,
                'Entry Price': row['Entry Price'],
                'Exit Price': row['Exit Price'],
                'Exit Reason': row['Exit Reason'],
                'Net PnL ($)': round(net_pnl, 2),
                'Running Equity ($)': round(current_equity, 2)
            })
            
        final_trades_df = pd.DataFrame(simulated_trades).sort_values(by='Exit Time').reset_index(drop=True)
        
        st.subheader("Performance Summary")
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
        
        st.subheader("Long vs. Short Breakdown")
        long_trades = final_trades_df[final_trades_df['Direction'] == 'LONG']
        short_trades = final_trades_df[final_trades_df['Direction'] == 'SHORT']
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("#### 🟢 Long Trades")
            if not long_trades.empty:
                l_pnl = long_trades['Net PnL ($)'].sum()
                l_win_rate = (long_trades['Net PnL ($)'] > 0).mean() * 100
                st.metric("Long Net PnL", f"${l_pnl:,.2f}")
                st.metric("Long Win Rate", f"{l_win_rate:.1f}%", f"{len(long_trades)} trades")
            else:
                st.info("No long trades taken.")
                
        with l_col2:
            st.markdown("#### 🔴 Short Trades")
            if not short_trades.empty:
                s_pnl = short_trades['Net PnL ($)'].sum()
                s_win_rate = (short_trades['Net PnL ($)'] > 0).mean() * 100
                st.metric("Short Net PnL", f"${s_pnl:,.2f}")
                st.metric("Short Win Rate", f"{s_win_rate:.1f}%", f"{len(short_trades)} trades")
            else:
                st.info("No short trades taken.")

        final_trades_df['Cumulative PnL'] = final_trades_df['Net PnL ($)'].cumsum()
        st.subheader("Strategy Equity Curve")
        st.line_chart(final_trades_df.set_index('Exit Time')['Cumulative PnL'])
        
        st.subheader("Master Trade Log")
        st.dataframe(final_trades_df, use_container_width=True)
else:
    st.info("👈 Please upload your SPY CSV file using the sidebar file uploader to begin.")
