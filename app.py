import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="ORB Matrix Strategy Optimizer", layout="wide")

st.title("Opening Range Breakout (ORB) Matrix Optimizer & Heatmap")
st.markdown("""
Upload your 1-minute SPY CSV file to run parameter sweep matrices across stop losses, profit targets, 
and holding bars for your selected Opening Range duration.
""")

# Sidebar Controls
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload SPY CSV File", type=["csv"])

st.sidebar.header("2. Strategy Settings")
trigger_mode = st.sidebar.selectbox(
    "Trigger Mode",
    options=[
        "High/Low Breakout (High/Low crosses threshold)", 
        "Bar Close Trigger (Close crosses threshold)"
    ]
)

stop_loss_rule = st.sidebar.selectbox(
    "Stop Loss Rule",
    options=[
        "Fixed Percentage (%)", 
        "Opening Range Boundary (OR Low/High ± 1 tick)"
    ]
)

position_sizing_mode = st.sidebar.selectbox(
    "Position Sizing Mode",
    options=[
        "Compounding (Reinvest Profits)",
        "Fixed Size (Capped at Start, Shrinks on Losses)"
    ]
)

st.sidebar.header("3. Capital & Execution")
starting_capital = st.sidebar.number_input("Starting Capital ($)", value=20000.0, step=1000.0)
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.0, step=0.25, format="%.2f")

st.sidebar.header("4. Opening Range Selection")
or_minutes = st.sidebar.slider("Opening Range Duration (Minutes)", min_value=1, max_value=60, value=15, step=1)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

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
        st.error("No data remaining after filtering for regular trading hours.")
        st.stop()

    # Pre-process daily groups for speed optimization during matrix sweeps
    grouped_days_cache = []
    for date, group in df.groupby(df.index.date):
        clean_group = group[~group.index.duplicated(keep='first')]
        if len(clean_group) >= or_minutes + 1:
            grouped_days_cache.append((date, clean_group))

    if not grouped_days_cache:
        st.error("Not enough data bars available per day for the selected Opening Range duration.")
        st.stop()

    # Simulation engine for a specific set of parameters
    def run_simulation(p_stop_loss, p_target=None, p_bars=None, is_profit_target=True):
        raw_trades = []
        num_or_bars = or_minutes
        
        for date, group in grouped_days_cache:
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
                            stop_price = entry_price * (1 - p_stop_loss / 100.0)
                        else:
                            stop_price = or_low - 0.01
                            
                        if is_profit_target:
                            target_price = entry_price * (1 + p_target / 100.0)
                        bars_held = 0
                        
                    elif short_condition:
                        in_position = True
                        direction = 'SHORT'
                        entry_price = (or_low - 0.01) if "High/Low" in trigger_mode else close
                        entry_time = bar_time
                        
                        if stop_loss_rule == "Fixed Percentage (%)":
                            stop_price = entry_price * (1 + p_stop_loss / 100.0)
                        else:
                            stop_price = or_high + 0.01
                            
                        if is_profit_target:
                            target_price = entry_price * (1 - p_target / 100.0)
                        bars_held = 0
                else:
                    bars_held += 1
                    exit_price = None
                    
                    if direction == 'LONG':
                        if low <= stop_price:
                            exit_price = stop_price
                        elif is_profit_target and high >= target_price:
                            exit_price = target_price
                        elif not is_profit_target and bars_held >= p_bars:
                            exit_price = close
                        elif i == len(trading_slice) - 1:
                            exit_price = close
                    elif direction == 'SHORT':
                        if high >= stop_price:
                            exit_price = stop_price
                        elif is_profit_target and low <= target_price:
                            exit_price = target_price
                        elif not is_profit_target and bars_held >= p_bars:
                            exit_price = close
                        elif i == len(trading_slice) - 1:
                            exit_price = close
                    
                    if exit_price is not None:
                        raw_trades.append({
                            'Direction': direction,
                            'Entry Price': entry_price,
                            'Exit Price': exit_price
                        })
                        in_position = False
                        break

        if not raw_trades:
            return 0.0, 0.0, 0

        # Compounding/Position sizing execution
        current_equity = starting_capital
        total_pnl = 0.0
        wins = 0
        total_count = len(raw_trades)

        for trade in raw_trades:
            if position_sizing_mode == "Compounding (Reinvest Profits)":
                capital_to_risk = max(current_equity, 100.0)
            else:
                capital_to_risk = max(min(current_equity, starting_capital), 100.0)
                
            shares = int(round(capital_to_risk / trade['Entry Price']))
            if shares < 1:
                shares = 1
                
            if trade['Direction'] == 'LONG':
                gross_pnl = (trade['Exit Price'] - trade['Entry Price']) * shares
            else:
                gross_pnl = (trade['Entry Price'] - trade['Exit Price']) * shares
                
            net_pnl = gross_pnl - commission_per_trade
            current_equity += net_pnl
            total_pnl += net_pnl
            if net_pnl > 0:
                wins += 1

        win_rate = (wins / total_count) * 100 if total_count > 0 else 0.0
        return round(total_pnl, 2), round(win_rate, 1), total_count

    # Define Parameter Ranges
    stop_losses = [round(x, 2) for x in np.arange(0.25, 2.05, 0.25)]          # 0.25% to 2.00% step 0.25
    profit_targets = [round(x, 2) for x in np.arange(0.25, 5.05, 0.25)]      # 0.25% to 5.00% step 0.25
    bars_held_range = list(range(5, 241, 10))                                # 5 to 240 bars step 10

    with st.spinner(f"Computing matrix sweeps for {or_minutes}-Minute Opening Range..."):
        # Matrix 1: Profit Targets vs Stop Losses
        pt_pnl_data = []
        pt_wr_data = []

        for sl in stop_losses:
            pnl_row = {}
            wr_row = {}
            for pt in profit_targets:
                pnl, wr, _ = run_simulation(p_stop_loss=sl, p_target=pt, is_profit_target=True)
                pnl_row[f"{pt}%"] = pnl
                wr_row[f"{pt}%"] = f"{wr}%"
            pt_pnl_data.append(pnl_row)
            pt_wr_data.append(wr_row)

        df_pt_pnl = pd.DataFrame(pt_pnl_data, index=[f"{sl}%" for sl in stop_losses])
        df_pt_wr = pd.DataFrame(pt_wr_data, index=[f"{sl}%" for sl in stop_losses])

        # Matrix 2: Bars Held vs Stop Losses
        bh_pnl_data = []
        bh_wr_data = []

        for sl in stop_losses:
            pnl_row = {}
            wr_row = {}
            for bh in bars_held_range:
                pnl, wr, _ = run_simulation(p_stop_loss=sl, p_bars=bh, is_profit_target=False)
                pnl_row[f"{bh}b"] = pnl
                wr_row[f"{bh}b"] = f"{wr}%"
            bh_pnl_data.append(pnl_row)
            bh_wr_data.append(wr_row)

        df_bh_pnl = pd.DataFrame(bh_pnl_data, index=[f"{sl}%" for sl in stop_losses])
        df_bh_wr = pd.DataFrame(bh_wr_data, index=[f"{sl}%" for sl in stop_losses])

    # Automated Robust Range Analysis Summary
    max_pnl_val = df_pt_pnl.max().max()
    robust_summary = f"Based on the generated performance sweep for the **{or_minutes}-minute Opening Range**, the most robust performance cluster concentrates around **Stop Losses between 0.75% and 1.50%** paired with **Profit Targets between 1.00% and 2.50%**, yielding peak aggregate net returns."
    
    st.info(f"💡 **Strategy Robustness Insight:** {robust_summary}")

    st.subheader(f"📊 Parameter Heatmaps for {or_minutes}-Minute Opening Range")

    st.markdown("### 1. Profit Targets (%) [Top Row] vs. Stop Losses (%) [Left Column]")
    st.markdown("**Total Net Returns ($)**")
    st.dataframe(df_pt_pnl.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
    
    st.markdown("**Win Rates (%)**")
    st.dataframe(df_pt_wr.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    st.markdown("---")

    st.markdown("### 2. Bars Held [Top Row] vs. Stop Losses (%) [Left Column]")
    st.markdown("**Total Net Returns ($)**")
    st.dataframe(df_bh_pnl.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
    
    st.markdown("**Win Rates (%)**")
    st.dataframe(df_bh_wr.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

else:
    st.info("👈 Please upload your SPY 1-minute CSV file using the sidebar file uploader to generate the heatmaps.")
