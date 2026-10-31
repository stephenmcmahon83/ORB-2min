import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Multi-Stock Intraday ORB Matrix Analyzer", layout="wide")

st.title("Opening Range Breakout (ORB) Matrix Analyzer")
st.markdown("""
Select your Opening Range duration below to instantly analyze performance matrices across 
**Stop Losses vs. Profit Targets** and **Stop Losses vs. Holding Bars**, utilizing per-ticker compounding equity.
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

# Dropdown for Opening Range Minutes (2 to 60, step 2)
st.sidebar.header("2. Strategy Parameter")
or_minutes_options = list(range(2, 62, 2))
selected_or_minutes = st.sidebar.selectbox("Opening Range Duration (Minutes)", options=or_minutes_options, value=30)

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

def run_matrix_simulation(all_data, p_or_mins, stop_loss_list, metric_mode_list, is_profit_target=True):
    # Pre-calculate raw trades for all days per ticker to speed up matrix loops
    ticker_daily_bars = {}
    for ticker, data in all_data.items():
        ticker_daily_bars[ticker] = list(data.groupby(data.index.date))
        
    matrix_results = []
    num_or_bars = p_or_mins // 2
    
    for sl in stop_loss_list:
        row_data = {}
        for m_val in metric_mode_list:
            raw_trades = []
            
            for ticker, grouped_days in ticker_daily_bars.items():
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
                                stop_price = entry_price * (1 - sl / 100.0)
                                if is_profit_target:
                                    target_price = entry_price * (1 + m_val / 100.0)
                                bars_held = 0
                            elif close < or_low:
                                in_position = True
                                direction = 'SHORT'
                                entry_price = close
                                entry_time = bar_time
                                stop_price = entry_price * (1 + sl / 100.0)
                                if is_profit_target:
                                    target_price = entry_price * (1 - m_val / 100.0)
                                bars_held = 0
                        else:
                            bars_held += 1
                            exit_price = None
                            
                            if direction == 'LONG':
                                if low <= stop_price:
                                    exit_price = stop_price
                                elif is_profit_target and high >= target_price:
                                    exit_price = target_price
                                elif not is_profit_target and bars_held >= m_val:
                                    exit_price = close
                                elif i == len(trading_slice) - 1:
                                    exit_price = close
                            elif direction == 'SHORT':
                                if high >= stop_price:
                                    exit_price = stop_price
                                elif is_profit_target and low <= target_price:
                                    exit_price = target_price
                                elif not is_profit_target and bars_held >= m_val:
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

            # Simulate PnL compounding per ticker
            if not raw_trades:
                row_data[m_val] = 0.0
            else:
                trades_df = pd.DataFrame(raw_trades).sort_values(by='Entry Time').reset_index(drop=True)
                ticker_equity = {t: initial_capital_per_ticker for t in tickers}
                total_net_pnl = 0.0
                
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
                    
                row_data[m_val] = round(total_net_pnl, 2)
                
        matrix_results.append(row_data)
        
    final_df = pd.DataFrame(matrix_results, index=stop_loss_list)
    return final_df

all_data = load_all_data(tickers)

if not all_data:
    st.error("No valid data retrieved for any of the specified tickers. Please verify the symbols.")
else:
    with st.spinner(f"Computing matrices for {selected_or_minutes}-minute Opening Range..."):
        # Define ranges as requested
        stop_losses = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
        profit_targets = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
        holding_bars_range = list(range(5, 245, 5)) # 5 to 240, step 5

        # Run Matrix 1: Profit Targets vs Stop Losses
        pt_matrix = run_matrix_simulation(all_data, selected_or_minutes, stop_losses, profit_targets, is_profit_target=True)
        pt_matrix.index.name = "Stop Loss (%)"

        # Run Matrix 2: Bars Held vs Stop Losses
        bar_matrix = run_matrix_simulation(all_data, selected_or_minutes, stop_losses, holding_bars_range, is_profit_target=False)
        bar_matrix.index.name = "Stop Loss (%)"

    # Display Results
    st.subheader(f"📊 Results for {selected_or_minutes}-Minute Opening Range")
    
    st.markdown("### 1. Profit Target (%) vs. Stop Loss (%) — Net PnL Matrix")
    st.markdown("Columns represent **Profit Targets (%)**, rows represent **Stop Losses (%)**.")
    st.dataframe(
        pt_matrix.style.background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### 2. Bars Held (5–240 bars) vs. Stop Loss (%) — Net PnL Matrix")
    st.markdown("Columns represent **Bars Held (2-min increments)**, rows represent **Stop Losses (%)**.")
    st.dataframe(
        bar_matrix.style.background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True
    )
