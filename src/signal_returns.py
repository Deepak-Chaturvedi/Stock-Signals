import sqlite3
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

HORIZONS = {
    "ret_1w": 7,
    "ret_2w": 14,
    "ret_1m": 30,
    "ret_3m": 90,
    "ret_6m": 180,
    "ret_1y": 365,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_signals_from_db(conn):
    """Load accumulation and EMA signals from the SQLite database."""
    signals_query = """
        SELECT DISTINCT Symbol, DATE(Date) AS Signal_date, 
        'Accumulation Signal' AS Signal_Type, 
        Close AS Signal_Price ,
        row_number() OVER (PARTITION BY DATE(Date) ORDER BY Date,Avg_Volume_Spike DESC, 
        AD_Slope desc) as Signal_Rank
        FROM SIGNAL_ACCUMULATION_STEADY 
        WHERE DATE(Date) >='2024-01-01'

        UNION ALL

        SELECT DISTINCT Symbol, DATE(date_2) AS Signal_date, 
            signal_2 AS Signal_Type, 
            Price_2 AS Signal_Price ,
            row_number() OVER (PARTITION BY DATE(date_2) ORDER BY 
            upFrom52wlow DESC, Vol2 DESC, DATE(Date_1) DESC , Vol1 DESC)
                    as Signal_Rank

        FROM SIGNAL_EMA_CROSS
        WHERE DATE(date_2) >='2024-01-01'
        order by   Signal_date DESC, Signal_Rank ASC
    """
    df = pd.read_sql_query(signals_query, conn)
    print(f"✅ Total signals loaded: {df.shape[0]}")
    return df

def merge_signals_with_prices(signals_df, stock_df):
    """Join signal data with stock prices to get post-signal price evolution."""
    format_stock_df = stock_df[["Symbol", "Date", "Close"]].copy()
    format_stock_df["Date"] = pd.to_datetime(format_stock_df["Date"])

    merged = pd.merge(signals_df, format_stock_df, on="Symbol", how="inner")
    merged = merged[["Symbol", "Signal_date", "Signal_Type", "Signal_Price","Signal_Rank", "Date", "Close"]]
    merged = merged.dropna()
    merged = merged[merged["Date"] >= pd.to_datetime(merged["Signal_date"])]
    merged = merged.sort_values(
        by=["Symbol", "Signal_Type", "Signal_date", "Date"],
        ascending=[True, True, False, False],
    )
    merged.rename(columns={"Close": "Price"}, inplace=True)

    print(f"✅ Total signals after merging: {merged.shape[0]}")
    return merged

def compute_returns(group):
    """Compute returns for a single (Symbol, Signal_Type, Signal_date, Signal_Price, Signal_Rank) group."""
    group = group.sort_values("Date").reset_index(drop=True)

    # ensure numeric columns before calculation
    group["Price"] = pd.to_numeric(group["Price"], errors="coerce")
    group["Signal_Price"] = pd.to_numeric(group["Signal_Price"], errors="coerce")

    sig_date = group.at[0, "Signal_date"]
    sig_price = group.at[0, "Signal_Price"]

    out = {
        "Symbol": group.at[0, "Symbol"],
        "Signal_Type": group.at[0, "Signal_Type"],
        "Signal_date": sig_date,
        "Signal_Price": sig_price,
        "Signal_Rank": group.at[0, "Signal_Rank"],
    }

    for name, days in HORIZONS.items():
        target = sig_date + pd.Timedelta(days=days)
        sel = group[group["Date"] >= target]

        if not sel.empty:
            p = pd.to_numeric(sel.iloc[0]["Price"], errors="coerce")
            d = sel.iloc[0]["Date"]

            if pd.notna(p) and pd.notna(sig_price) and sig_price != 0:
                out[name] = round((p / sig_price - 1) * 100, 2)
            else:
                out[name] = pd.NA

            out[f"{name}_price"] = p
            out[f"{name}_date"] = d
        else:
            out[name] = pd.NA
            out[f"{name}_price"] = pd.NA
            out[f"{name}_date"] = pd.NaT

    last_row = group.iloc[-1]
    last_price = pd.to_numeric(last_row["Price"], errors="coerce")
    if pd.notna(last_price) and pd.notna(sig_price) and sig_price != 0:
        out["ret_sinceSignal"] = round((last_price / sig_price - 1) * 100, 2)
    else:
        out["ret_sinceSignal"] = pd.NA

    out["current_price"] = last_price
    out["current_date"] = last_row["Date"]

    return pd.Series(out)

def calculate_returns(merged_df):
    """Apply return computation across all signal groups."""
    merged_df["Signal_date"] = pd.to_datetime(merged_df["Signal_date"])
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])

    returns_df = (
        merged_df.groupby(["Symbol", "Signal_Type", "Signal_date", "Signal_Price","Signal_Rank"], sort=False)
        .apply(compute_returns)
        .reset_index(drop=True)
    )

    # Convert numeric columns to ensure no text values
    ret_cols = [c for c in returns_df.columns if c.startswith("ret_") and not c.endswith("_price") and not c.endswith("_date")]
    for c in ret_cols:
        returns_df[c] = pd.to_numeric(returns_df[c], errors="coerce").round(2)

    price_cols = [c for c in returns_df.columns if c.endswith("_price")]
    for c in price_cols:
        returns_df[c] = pd.to_numeric(returns_df[c], errors="coerce")

    returns_df["Signal_Price"] = pd.to_numeric(returns_df["Signal_Price"], errors="coerce")

    # Calculate percentage returns based on *_price and Signal_Price
    for horizon in HORIZONS.keys():
        returns_df[f"{horizon}_perc"] = returns_df.apply(
            lambda row: round((row[f"{horizon}_price"] / row["Signal_Price"] - 1) * 100, 2)
            if pd.notna(row[f"{horizon}_price"]) and pd.notna(row["Signal_Price"]) and row["Signal_Price"] != 0
            else pd.NA,
            axis=1
        )

    # ret_sinceSignal_perc
    returns_df["ret_sinceSignal_perc"] = returns_df.apply(
        lambda row: round((row["current_price"] / row["Signal_Price"] - 1) * 100, 2)
        if pd.notna(row["current_price"]) and pd.notna(row["Signal_Price"]) and row["Signal_Price"] != 0
        else pd.NA,
        axis=1
    )


    return returns_df

def get_new_records_only(conn, new_df):
    """
    Compare new_df with existing SIGNAL_RETURNS table and return
    only records that are not already in the table.
    """
    try:
        existing = pd.read_sql_query("SELECT * FROM SIGNAL_RETURNS", conn)
        print(f"📦 Existing SIGNAL_RETURNS records: {existing.shape[0]}")
    except Exception:
        print("⚠️ No existing SIGNAL_RETURNS table found. Treating all new_df as new.")
        existing = pd.DataFrame(columns=new_df.columns)

    # Ensure consistent datetime types
    new_df['Signal_date'] = pd.to_datetime(new_df['Signal_date'])
    if not existing.empty:
        existing['Signal_date'] = pd.to_datetime(existing['Signal_date'])
        existing.sort_values(by=["Symbol", "Signal_Type", "Signal_date", "Signal_Rank","current_date"], inplace=True)
        existing = existing.drop_duplicates(subset=["Symbol", "Signal_Type", "Signal_date","Signal_Rank"], keep="last")

    # Identify new rows not in existing
    if existing.empty:
        new_records = new_df.copy()
    else:
        merged = new_df.merge(
            existing[["Symbol", "Signal_Type", "Signal_date","Signal_Rank"]],
            on=["Symbol", "Signal_Type", "Signal_date","Signal_Rank"],
            how="left",
            indicator=True
        )
        new_records = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        # Keep only columns from new_df
        new_records = new_records[new_df.columns]

    print(f"✅ New unique records found: {new_records.shape[0]}")
    return new_records






# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_signal_returns(db_path, stock_df):
    """
    Main function to compute and update signal returns.

    Args:
        db_path (str): Path to SQLite database.
        stock_df (pd.DataFrame): Stock price data with columns [Symbol, Date, Close].

    Returns:
        tuple: (status: bool, result_df: pd.DataFrame)
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)

        signals_df = load_signals_from_db(conn)
        merged_df = merge_signals_with_prices(signals_df, stock_df)
        returns_df = calculate_returns(merged_df)

        # combined_df = combine_with_existing(conn, returns_df)
        new_records = get_new_records_only(conn, returns_df)
        
        print("🧩 Columns in signals_df before saving:", signals_df.columns.tolist())
        print("🧩 Columns in merged_df before saving:", merged_df.columns.tolist())
        print("🧩 Columns in new_records before saving:", new_records.columns.tolist())

        try:
            new_records.to_sql('SIGNAL_RETURNS', conn, if_exists='append', index=False)
            print("💾 SIGNAL_RETURNS table updated successfully.")
        except Exception as e:
            conn.rollback()
            print("Error:", e)

        # combined_df.to_sql("SIGNAL_RETURNS", conn, if_exists="replace", index=False)
        

        return True, new_records

    except Exception as e:
        print(f"❌ Error during signal returns generation: {e}")
        return False, None

    finally:
        if conn:
            conn.close()
            print("🔒 Database connection closed.")
