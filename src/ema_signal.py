# ============================================================
# src/ema_signal.py
# ============================================================
import pandas as pd
import numpy as np
import datetime
import time
import yfinance as yf


# ============================================================
# EMA SIGNAL GENERATION
# ============================================================
def get_ema_signal(data, start_dt, end_dt, ema1, ema2, HighSignal, LowSignal, vol_thresh=1.5):
    """Detect EMA crossovers (Golden/Death) with volume filters."""
    mask_period = (data["Date"] >= start_dt) & (data["Date"] <= end_dt)
    df = data.loc[mask_period].copy()

    # Find tickers with both conditions (ema1>ema2 and ema1<ema2 in period)
    df_gc1 = df.loc[df[ema1] > df[ema2], "Ticker"].unique()
    df_gc2 = df.loc[df[ema1] < df[ema2], "Ticker"].unique()
    cross_tickers = np.intersect1d(df_gc1, df_gc2)

    signals = []

    for ticker in cross_tickers:
        stock = df[df["Ticker"] == ticker].reset_index(drop=True)
        cross_high = stock[stock[ema1] > stock[ema2]].index
        cross_low = stock[stock[ema1] < stock[ema2]].index

        for i in range(1, len(stock)):
            if (i - 1 in cross_low) and (i in cross_high):
                signals.append([ticker, HighSignal, stock.loc[i, "Date"], stock.loc[i, "Close"],
                                stock.loc[i, "Volume"], stock.loc[i, "7dv"], stock.loc[i, "30dv"]])
            elif (i - 1 in cross_high) and (i in cross_low):
                signals.append([ticker, LowSignal, stock.loc[i, "Date"], stock.loc[i, "Close"],
                                stock.loc[i, "Volume"], stock.loc[i, "7dv"], stock.loc[i, "30dv"]])

    if not signals:
        print("No crossovers found.")
        return pd.DataFrame()

    signal_df = pd.DataFrame(signals, columns=["Ticker", "signal", "Date", "Price", "Volume", "7dAvgVol", "30dAvgVol"])
    signal_df["Symbol"] = signal_df["Ticker"].str.replace(".NS", "", regex=False)

    # Volume filters
    signal_df["7dratio"] = signal_df["Volume"] / signal_df["7dAvgVol"]
    signal_df["30dratio"] = signal_df["7dAvgVol"] / signal_df["30dAvgVol"]
    signal_df["volume_change"] = signal_df[["7dratio", "30dratio"]].max(axis=1)
    signal_df = signal_df[signal_df["30dAvgVol"] >= 1000]
    signal_df = signal_df[signal_df["volume_change"] > vol_thresh]
    signal_df = signal_df[signal_df["Price"] > 10]

    signal_df.sort_values(by=["Date", "volume_change"], ascending=False, inplace=True)
    print(f"{signal_df.Ticker.nunique()} tickers with {ema1}/{ema2} crossover between {start_dt}–{end_dt}")

    return signal_df


# ============================================================
# CHANDE MOMENTUM OSCILLATOR (CMO)
# ============================================================
def calculate_cmo(df, period=50, price_var="Close"):
    """Compute Chande Momentum Oscillator and 7-day smoothed CMO."""
    df = df.copy()
    df["Change"] = df[price_var].diff()
    df["Gain"] = df["Change"].clip(lower=0)
    df["Loss"] = -df["Change"].clip(upper=0)
    df["Sum_Gain"] = df["Gain"].rolling(window=period).sum()
    df["Sum_Loss"] = df["Loss"].rolling(window=period).sum()
    df["CMO"] = 100 * (df["Sum_Gain"] - df["Sum_Loss"]) / (df["Sum_Gain"] + df["Sum_Loss"])
    df["7dCMO"] = df.groupby("Ticker")["CMO"].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    return df.drop(columns=["Change", "Gain", "Loss", "Sum_Gain", "Sum_Loss"])


# ============================================================
# ACCUMULATION / DISTRIBUTION (AD)
# ============================================================
def calculate_ad(df):
    """Calculate AD and normalized 50-day AD."""
    df = df.copy()
    df["AD_raw"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]) * df["Volume"]
    df["AD"] = (df["AD_raw"] / df["Volume"]).round(2)
    df["AD_50"] = df.groupby("Ticker")["AD_raw"].transform(lambda x: x.rolling(50, min_periods=1).sum())
    df["AD_50"] = (df["AD_50"] / df.groupby("Ticker")["AD_50"].transform(lambda x: x.abs().max())).round(2)
    df["Symbol"] = df["Ticker"].str.replace(".NS", "", regex=False)
    return df[["Date", "Symbol", "AD", "AD_50"]]


# ============================================================
# STOCK METADATA FETCHER
# ============================================================
def get_stock_metadata(symbols, sleep_time=2):
    """Fetch stock metadata from Yahoo Finance."""
    fields = [
        "symbol", "currentPrice", "marketCap", "52WeekChange", "debtToEquity", "priceToBook",
        "trailingPE", "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "trailingEps",
        "priceToSalesTrailing12Months", "totalCashPerShare", "beta"
    ]
    stock_data = []

    for sym in symbols:
        try:
            info = yf.Ticker(sym + ".NS").info
            stock_data.append({field: info.get(field) for field in fields})
            time.sleep(sleep_time)
        except Exception as e:
            print(f"⚠️ Error fetching {sym}: {e}")

    df_meta = pd.DataFrame(stock_data).drop_duplicates()
    df_meta["SYMBOL"] = df_meta["symbol"].str.replace(".NS", "", regex=False)
    df_meta["update_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Derived columns
    df_meta["debtToEquity"] = df_meta["debtToEquity"] / 100
    df_meta["MCapCrore"] = (df_meta["marketCap"] / 1e7).round(2)
    df_meta["upFrom52wlow"] = (df_meta["currentPrice"] - df_meta["fiftyTwoWeekLow"]) / (
        df_meta["fiftyTwoWeekHigh"] - df_meta["fiftyTwoWeekLow"]
    )

    bins = [-float("inf"), 500, 1000, 5000, 20000, float("inf")]
    labels = ["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]
    df_meta["Capitalization"] = pd.cut(df_meta["MCapCrore"], bins=bins, labels=labels)

    df_meta["PricetoCash"] = df_meta["currentPrice"] / df_meta["totalCashPerShare"]
    print(f"✅ Metadata fetched for {len(df_meta)} stocks")
    return df_meta


# ============================================================
# FINAL WRAPPER FUNCTION
# ============================================================
def generate_ema_signals(df, analysis_period=60, vol_thresh=1.5, output_period=30): # fno=None)
    """
    Full EMA signal generation pipeline.
    Returns: long_stocks, short_stocks, ema_df_final
    """
    start_dt = (datetime.datetime.today() - datetime.timedelta(days=analysis_period)).strftime("%Y-%m-%d")
    end_dt = pd.Timestamp.today().strftime("%Y-%m-%d")

    # --- Step 1: Get signals ---
    price_cross = get_ema_signal(df, start_dt, end_dt, "Close", "200ema", "Price_gt_200ema", "Price_lt_200ema", 2)
    golden_cross = get_ema_signal(df, start_dt, end_dt, "50ema", "200ema", "Golden cross", "Death cross", vol_thresh)

    # --- Step 2: Momentum + AD ---
    cmo_df = calculate_cmo(df)
    ad_df = calculate_ad(df)

    for s in [price_cross, golden_cross]:
        s.merge(cmo_df[["Ticker", "Date", "CMO"]], on=["Ticker", "Date"], how="left")

    ema_signal = pd.merge(price_cross, golden_cross, on="Symbol", how="outer", suffixes=("_1", "_2"))
    ema_signal = ema_signal.merge(ad_df, left_on=["Date_2", "Symbol"], right_on=["Date", "Symbol"], how="left")
    ema_signal.drop(columns=["Date"], inplace=True)

    # --- Step 3: Metadata ---
    meta_df = get_stock_metadata(ema_signal.Symbol.unique())
    join_vars = [
        "SYMBOL", "currentPrice", "Capitalization", "MCapCrore", "upFrom52wlow", "debtToEquity",
        "priceToBook", "trailingPE", "trailingEps", "priceToSalesTrailing12Months",
        "PricetoCash", "update_date" , "beta"
    ]
    ema_df = ema_signal.merge(meta_df[join_vars], how="left", left_on="Symbol", right_on="SYMBOL")

    # --- Step 4: Filtering ---
    # ema_df = ema_df.dropna(subset=["trailingEps", "marketCap"])
    ema_df = ema_df[(ema_df["trailingEps"] > 0) & (ema_df["debtToEquity"] <= 2.0) & (ema_df["MCapCrore"] >= 200)]
    ema_df = ema_df[ema_df["trailingPE"] <= 100].drop_duplicates()

    # --- Step 5: Add FnO flag ---
    # if fno is not None:
    #     ema_df = ema_df.merge(fno[["Symbol"]], how="left", on="Symbol", indicator=True)
    #     ema_df["FnO"] = np.where(ema_df["_merge"] == "both", "Yes", "No")
    #     ema_df.drop(columns=["_merge"], inplace=True)
    # else:
    #     ema_df["FnO"] = "No"

    # --- Step 6: Split final sets ---
    ema_df["Date_1"] = ema_df["Date_1"].dt.strftime("%Y-%m-%d")
    ema_df["Date_2"] = ema_df["Date_2"].dt.strftime("%Y-%m-%d")

    cutoff = (datetime.datetime.today() - datetime.timedelta(days=output_period)).strftime("%Y-%m-%d")
    rec_df = ema_df[(ema_df["Date_1"] >= cutoff) | (ema_df["Date_2"] >= cutoff)]

    long_stocks = rec_df[(rec_df["signal_2"] == "Golden cross") | (rec_df["signal_1"] == "Price_gt_200ema")]
    short_stocks = rec_df[ #(rec_df["FnO"] == "Yes") &
                          ((rec_df["signal_2"] == "Death cross") | (rec_df["signal_1"] == "Price_lt_200ema"))]

    print(f"✅ Final signals: {len(long_stocks)} long | {len(short_stocks)} short")
    return long_stocks, short_stocks, ema_df


# ============================================================
# EXAMPLE USAGE (for notebook)
# ============================================================
if __name__ == "__main__":
    print("Run this module from your notebook like:")
    print("from src.ema_signal import generate_ema_signals")
