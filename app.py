import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Multi-File Portfolio ORB Backtester", layout="wide")

st.title("Multi-Ticker Portfolio Intraday Strategy Backtester")
st.markdown("""
Upload multiple 1-minute CSV files (one for each ticker, e.g., `SPY.csv`, `QQQ.csv`). The app runs 
independent gap-filtered backtests on each ticker, divides your total starting capital across positions, 
and aggregates the portfolio results chronologically.
""")

# Sidebar Controls
st.sidebar.header("1. Data Upload (Multiple Files)")
uploaded_files = st.sidebar.file_uploader(
    "Upload 1-Min CSV Files", 
    type=["csv"], 
    accept_multiple_files=True
)

st.sidebar.header("2. Portfolio Capital & Sizing")
total_starting_capital = st.sidebar.number_input("Total Portfolio Starting Capital ($)", value=20000.0, step=1000.0)
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.25, format="%.2f")

position_sizing_mode = st.sidebar.selectbox(
    "Position Sizing Mode",
    options=[
        "Compounding (Reinvest Profits per Ticker)",
        "Fixed Size (Capped at Initial Allocation, Shrinks on Losses)"
    ]
)

st.sidebar.header("3. Strategy & Trigger Mode")
strategy_type = st.sidebar.selectbox(
    "Choose Strategy", 
    options=["Opening Range Breakout (ORB)", "Open-Percent Threshold Strategy"]
)

trigger_mode = st.sidebar.selectbox(
    "Trigger Mode",
    options=[
        "High/Low Breakout (High/Low crosses threshold)", 
        "Bar Close Trigger (Close crosses threshold)"
    ]
)

# Strategy Parameters
st.sidebar.header("4. Strategy Parameters")
if strategy_type == "Opening Range Breakout (ORB)":
    or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=1, max_value=60, value=15, step=1)
    exit_mode = st.sidebar.selectbox("Exit Mode", options=["Profit Target (%)", "Hold for X Bars"])
    
    stop_loss_rule = st.sidebar.selectbox(
        "Stop Loss Rule", 
        options=["Fixed Percentage (%)", "Opening Range Boundary (OR Low/High ± 1 tick)"]
    )
    if stop_loss_rule == "Fixed Percentage (%)":
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.25, max_value=3.00, value=0.50, step=0.25)
    else:
        stop_loss_pct = 0.0

    if exit_mode == "Profit Target (%)":
        profit_target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.25, max_value=3.00, value=1.00, step=0.25)
        holding_bars = 0
    else:
        profit_target_pct = 0.0
        holding_bars = st.sidebar.slider("Holding Bars (1-min intervals)", min_value=1, max_value=400, value=10, step=1)
else:
    trade_direction = st.sidebar.selectbox("Trade Direction", options=["Both (Long & Short)", "Long Only", "Short Only"])
    entry_threshold_pct = st.sidebar.slider("Entry Threshold (% from Open)", min_value=0.05, max_value=2.0, value=0.25, step=0.05)
    profit_target_pct = st.sidebar.slider("Profit Target (%)", min_value=0.25, max_value=3.00, value=1.00, step=0.25)
    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.25, max_value=3.00, value=0.50, step=0.25)
    stop_loss_rule = "Fixed Percentage (%)"

if uploaded_files:
    num_tickers = len(uploaded_files)
    initial_capital_per_ticker = total_starting_capital / max(num_tickers, 1)
    st.sidebar.markdown(f"**Active Tickers Loaded:** {num_tickers}")
    st.sidebar.markdown(f"**Initial Capital per Ticker:** ${initial_capital_per_ticker:,.2f}")

    all_ticker_data = {}
    
    # Load and parse each uploaded file
    for file in uploaded_files:
        ticker_name = file.name.split('.')[0].upper()
        try:
            df = pd.read_csv(file, sep=None, engine='python')
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

            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York')
            else:
                try:
                    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
                except Exception:
                    pass 
                
            df = df.between_time('09:30', '16:00')
            if not df.empty:
                all_ticker_data[ticker_name] = df
        except Exception as e:
            st.error(f"Error parsing file {file.name}: {e}")

    if not all_ticker_data:
        st.warning("No valid data loaded from the uploaded files.")
    else:
        raw_trades = []

        # Run independent backtest per ticker
        for ticker, df in all_ticker_data.items():
            grouped_days = df.groupby(df.index.date)
            
            for date, group in grouped_days:
                group = group[~group.index.duplicated(keep='first')]
                min_required_bars = (or_minutes + 1) if strategy_type == "Opening Range Breakout (ORB)" else 2
                if len(group) < min_required_bars:
                    continue
                
                if strategy_type == "Opening Range Breakout (ORB)":
                    num_or_bars = or_minutes
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
                        high = bar['High']
                        low = bar['Low']
                        close = bar['Close']
                        
                        if not in_position:
                            if trigger_mode == "High/Low Breakout (High/Low crosses threshold)":
                                long_condition = (high >= or_high + 0.01)
                                short_condition = (low <= or_low - 0.01)
                            else:
                                long_condition = (close > or_high)
                                short_condition = (close < or_low)

                            if long_condition:
                                in_position = True
                                direction = 'LONG'
                                entry_price = (or_high + 0.01) if "High/Low" in trigger_mode else close
                                entry_time = bar_time
                                
                                if stop_loss_rule == "Fixed Percentage (%)":
                                    stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                                else:
                                    stop_price = or_low - 0.01
                                    
                                if exit_mode == "Profit Target (%)":
                                    target_price = entry_price * (1 + profit_target_pct / 100.0)
                                bars_held = 0
                                
                            elif short_condition:
                                in_position = True
                                direction = 'SHORT'
                                entry_price = (or_low - 0.01) if "High/Low" in trigger_mode else close
                                entry_time = bar_time
                                
                                if stop_loss_rule == "Fixed Percentage (%)":
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
                    # Open-Percent Threshold Strategy
                    day_open = group.iloc[0]['Open']
                    in_position = False
                    entry_price = 0.0
                    entry_time = None
                    direction = None
                    target_price = 0.0
                    stop_price = 0.0
                    
                    long_trigger_val = day_open * (1 + entry_threshold_pct / 100.0)
                    short_trigger_val = day_open * (1 - entry_threshold_pct / 100.0)
                    
                    for i in range(len(group)):
                        bar = group.iloc[i]
                        bar_time = group.index[i]
                        high = bar['High']
                        low = bar['Low']
                        close = bar['Close']
                        
                        if not in_position:
                            if trigger_mode == "High/Low Breakout (High/Low crosses threshold)":
                                long_cond = (high >= long_trigger_val)
                                short_cond = (low <= short_trigger_val)
                            else:
                                long_cond = (close >= long_trigger_val)
                                short_cond = (close <= short_trigger_val)

                            if trade_direction in ["Both (Long & Short)", "Long Only"] and long_cond:
                                in_position = True
                                direction = 'LONG'
                                entry_price = long_trigger_val if "High/Low" in trigger_mode else close
                                entry_time = bar_time
                                target_price = entry_price * (1 + profit_target_pct / 100.0)
                                stop_price = entry_price * (1 - stop_loss_pct / 100.0)
                            elif trade_direction in ["Both (Long & Short)", "Short Only"] and short_cond:
                                in_position = True
                                direction = 'SHORT'
                                entry_price = short_trigger_val if "High/Low" in trigger_mode else close
                                entry_time = bar_time
                                target_price = entry_price * (1 - profit_target_pct / 100.0)
                                stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                        else:
                            exit_price = None
                            exit_reason = None
                            
                            if direction == 'LONG':
                                if low <= stop_price:
                                    exit_price = stop_price
                                    exit_reason = 'Stop Loss'
                                elif high >= target_price:
                                    exit_price = target_price
                                    exit_reason = 'Profit Target'
                                elif i == len(group) - 1:
                                    exit_price = close
                                    exit_reason = 'EOD Close'
                            elif direction == 'SHORT':
                                if high >= stop_price:
                                    exit_price = stop_price
                                    exit_reason = 'Stop Loss'
                                elif low <= target_price:
                                    exit_price = target_price
                                    exit_reason = 'Profit Target'
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
            st.warning("No trades were triggered across any of the uploaded files using current parameters.")
        else:
            # Sort all trades chronologically across all tickers
            trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
            
            # Maintain independent capital balance per ticker
            ticker_equity = {t: initial_capital_per_ticker for t in all_ticker_data.keys()}
            simulated_trades = []
            
            for _, row in trades_df.iterrows():
                ticker = row['Ticker']
                current_ticker_capital = ticker_equity.get(ticker, initial_capital_per_ticker)
                
                if position_sizing_mode == "Compounding (Reinvest Profits per Ticker)":
                    capital_to_risk = max(current_ticker_capital, 100.0)
                else:
                    capital_to_risk = max(min(current_ticker_capital, initial_capital_per_ticker), 100.0)
                    
                shares = int(round(capital_to_risk / row['Entry Price']))
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
                    'Allocation Size ($)': round(capital_to_risk, 2),
                    'Shares': shares,
                    'Entry Price': row['Entry Price'],
                    'Exit Price': row['Exit Price'],
                    'Exit Reason': row['Exit Reason'],
                    'Net PnL ($)': round(net_pnl, 2),
                    'Ticker Balance After ($)': round(ticker_equity[ticker], 2)
                })
                
            final_trades_df = pd.DataFrame(simulated_trades).sort_values(by='Exit Time').reset_index(drop=True)
            
            # Portfolio Summary Metrics
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
            col4.metric("Wins / Losses", f"{winning_trades} / {losing_trades}")
            
            # Breakdown by Ticker
            st.subheader("Performance Breakdown by Ticker")
            ticker_summary = final_trades_df.groupby('Ticker').agg(
                Total_Trades=('Net PnL ($)', 'count'),
                Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean() * 100:.1f}%"),
                Net_PnL=('Net PnL ($)', 'sum'),
                Ending_Balance=('Ticker Balance After ($)', 'last')
            ).reset_index()
            ticker_summary['Net_PnL'] = ticker_summary['Net_PnL'].apply(lambda x: f"${x:,.2f}")
            ticker_summary['Ending_Balance'] = ticker_summary['Ending_Balance'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(ticker_summary, use_container_width=True)

            # Long vs Short Breakdown
            st.subheader("Long vs. Short Breakdown")
            long_trades = final_trades_df[final_trades_df['Direction'] == 'LONG']
            short_trades = final_trades_df[final_trades_df['Direction'] == 'SHORT']
            
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.markdown("#### 🟢 Portfolio Longs")
                if not long_trades.empty:
                    l_pnl = long_trades['Net PnL ($)'].sum()
                    l_win_rate = (long_trades['Net PnL ($)'] > 0).mean() * 100
                    st.metric("Long Net PnL", f"${l_pnl:,.2f}")
                    st.metric("Long Win Rate", f"{l_win_rate:.1f}%", f"{len(long_trades)} trades")
                else:
                    st.info("No long trades taken.")
                    
            with l_col2:
                st.markdown("#### 🔴 Portfolio Shorts")
                if not short_trades.empty:
                    s_pnl = short_trades['Net PnL ($)'].sum()
                    s_win_rate = (short_trades['Net PnL ($)'] > 0).mean() * 100
                    st.metric("Short Net PnL", f"${s_pnl:,.2f}")
                    st.metric("Short Win Rate", f"{s_win_rate:.1f}%", f"{len(short_trades)} trades")
                else:
                    st.info("No short trades taken.")

            # Combined Portfolio Equity Curve
            # Calculate cumulative portfolio PnL across all tickers chronologically
            final_trades_df['Cumulative Portfolio PnL'] = final_trades_df['Net PnL ($)'].cumsum()
            st.subheader("Combined Portfolio Equity Curve")
            st.line_chart(final_trades_df.set_index('Exit Time')['Cumulative Portfolio PnL'])
            
            # Master Trade Log
            st.subheader("Master Portfolio Trade Log")
            st.dataframe(final_trades_df, use_container_width=True)
else:
    st.info("👈 Please upload your 1-minute CSV files (e.g., SPY.csv, QQQ.csv) using the sidebar uploader to begin portfolio backtesting.")
