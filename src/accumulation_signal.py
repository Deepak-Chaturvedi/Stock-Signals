# src/accumulation_signal.py

import pandas as pd
import numpy as np
from scipy.stats import linregress
import sqlite3

def calculate_ad_line(df):
    """Calculate Accumulation/Distribution Line (cumulative)."""
    mf_multiplier = ((2 * df['Close'] - df['High'] - df['Low']) /
                     (df['High'] - df['Low']).replace(0, np.nan))
    mf_volume = mf_multiplier * df['Volume']
    return mf_volume.cumsum()


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_gradual_accumulation(df_all,
                                 ad_window=20,
                                 slope_threshold=0.01,
                                 price_ma_ratio=0.8,
                                 rsi_threshold=50,
                                 volume_spike_threshold=2.5):
    """
    Detect gradual accumulation with A/D slope, price above EMA, RSI, and volume filters.
    """
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    results = []

    for symbol, df in df_all.groupby('Symbol'):
        df = df.sort_values('Date').copy()

        if len(df) < ad_window + 10:
            continue

        df['ad_line'] = calculate_ad_line(df)

        def slope_fn(series):
            x = np.arange(len(series))
            y = series.values
            return linregress(x, y).slope

        df['ad_slope'] = df['ad_line'].rolling(ad_window).apply(slope_fn, raw=False)
        df['above_ema'] = df['Close'] > df['20ema']
        df['above_ema_count'] = df['above_ema'].rolling(ad_window).sum()
        df['percent_above_ema'] = df['above_ema_count'] / ad_window

        df['avg_volume'] = df['Volume'].rolling(ad_window).mean()
        df['volume_spike_ratio'] = df['Volume'] / df['avg_volume']
        df['avg_spike'] = df['volume_spike_ratio'].rolling(ad_window).mean()

        df['rsi'] = calculate_rsi(df['Close'])

        df['accumulating'] = (
            (df['ad_slope'] > slope_threshold) &
            (df['percent_above_ema'] >= price_ma_ratio) &
            (df['avg_spike'] <= volume_spike_threshold) &
            (df['rsi'] > rsi_threshold)
        )

        latest = df.iloc[-1]
        results.append({
            'Symbol': symbol,
            'Date': latest['Date'],
            'AD_Slope': round(latest['ad_slope'], 4),
            'Above_20EMA_%': round(latest['percent_above_ema'] * 100, 1),
            'Avg_Volume_Spike': round(latest['avg_spike'], 2),
            'RSI': round(latest['rsi'], 2),
            'Accumulating': bool(latest['accumulating']),
            '20ema': latest['20ema'],
            '50ema': latest['50ema'],
            '200ema': latest['200ema'],
            'Close': latest['Close'],
            '7dv': latest['7dv'],
            '30dv': latest['30dv'],
        })

    return pd.DataFrame(results).sort_values(by='Accumulating', ascending=False)


def filter_accumulation(accumulation_df):
    """Filter stocks that meet additional entry criteria."""
    accum_df = accumulation_df[
        (accumulation_df['20ema'] > accumulation_df['50ema']) &
        (accumulation_df['50ema'] < accumulation_df['200ema']) &
        (accumulation_df['Close'] > accumulation_df['50ema']) &
        (accumulation_df['7dv'] >= 1.5 * accumulation_df['30dv'])
    ]
    return accum_df.sort_values('AD_Slope', ascending=False).reset_index(drop=True)


def save_accumulation_to_db(accum_df, db_path, table_name='SIGNAL_ACCUMULATION_STEADY'):
    """
    Append accumulation DataFrame to SQLite database,
    remove duplicates based on ['Symbol', 'Date'],
    sort by Date descending,
    and overwrite the table.
    """

    conn = sqlite3.connect(db_path)
    success = False  # default status

    try:
        # Step 1: Load existing table
        try:
            existing_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            existing_df = pd.DataFrame()  # Table might not exist yet

        # Step 2: Normalize new data
        if "Date" in accum_df.columns:
            accum_df['Date'] = pd.to_datetime(accum_df['Date'], errors='coerce')
        if "Symbol" in accum_df.columns:
            accum_df['Symbol'] = accum_df['Symbol'].astype(str).str.strip()

        # Step 3: Normalize existing data
        if not existing_df.empty:
            if "Date" in existing_df.columns:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
            if "Symbol" in existing_df.columns:
                existing_df['Symbol'] = existing_df['Symbol'].astype(str).str.strip()

        # Step 4: Append new data
        combined_df = pd.concat([existing_df, accum_df], ignore_index=True)

        # Step 5: Sort by Date descending
        if "Date" in combined_df.columns:
            combined_df.sort_values(by="Date", ascending=False, inplace=True)

        # Step 6: Remove duplicates based on Symbol and Date
        combined_df.drop_duplicates(subset=['Symbol', 'Date'], keep='first', inplace=True)

        # Step 7: Overwrite table
        combined_df.to_sql(table_name, conn, if_exists='replace', index=False)
        success = True  # operation successful
        
    except Exception as e:
        print(f"Error saving accumulation data: {e}")
        success = False
        raise

    finally:
        conn.close()

    return success

