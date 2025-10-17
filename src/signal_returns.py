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
                        Close AS Signal_Price 
        FROM SIGNAL_ACCUMULATION_STEADY 
        WHERE DATE(Date) >='2024-01-01'
        UNION
        SELECT DISTINCT Symbol, DATE(date_2) AS Signal_date, 
                        signal_2 AS Signal_Type, 
                        Price_2 AS Signal_Price 
        FROM SIGNAL_EMA_CROSS
        WHERE DATE(date_2) >='2024-01-01'
    """
    df = pd.read_sql_query(signals_query, conn)
    print(f"✅ Total signals loaded: {df.shape[0]}")
    return df

def merge_signals_with_prices(signals_df, stock_df):
    """Join signal data with stock prices to get post-signal price evolution."""
    format_stock_df = stock_df[["Symbol", "Date", "Close"]].copy()
    format_stock_df["Date"] = pd.to_datetime(format_stock_df["Date"])

    merged = pd.merge(signals_df, format_stock_df, on="Symbol", how="inner")
    merged = merged[["Symbol", "Signal_date", "Signal_Type", "Signal_Price", "Date", "Close"]]
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
    """Compute returns for a single (Symbol, Signal_Type, Signal_date, Signal_Price) group."""
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
        merged_df.groupby(["Symbol", "Signal_Type", "Signal_date", "Signal_Price"], sort=False)
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

# def combine_with_existing(conn, new_df):
#     """
#     Merge new returns data with existing SIGNAL_RETURNS table,
#     keep only latest current_date per signal, and print summary.
#     """
#     try:
#         existing = pd.read_sql_query("SELECT * FROM SIGNAL_RETURNS", conn)
#         print(f"📦 Existing SIGNAL_RETURNS records: {existing.shape[0]}")
#     except Exception:
#         print("⚠️ No existing SIGNAL_RETURNS table found. Creating a new one.")
#         existing = pd.DataFrame(columns=new_df.columns)

#     before_merge = existing.shape[0]

#     combined = pd.concat([existing, new_df], ignore_index=True)
#     before_dedup = combined.shape[0]

#     # Keep only the latest current_date per signal
#     combined.sort_values(by=["Symbol", "Signal_Type", "Signal_date", "current_date"], inplace=True)
#     combined = combined.drop_duplicates(subset=["Symbol", "Signal_Type", "Signal_date"], keep="last")

#     after_dedup = combined.shape[0]

#     added_records = after_dedup - before_merge
#     dropped_duplicates = before_dedup - after_dedup

#     print(f"✅ SIGNAL_RETURNS summary:")
#     print(f"   • Existing records      : {before_merge}")
#     print(f"   • Newly added (unique)  : {added_records}")
#     print(f"   • Duplicates removed    : {dropped_duplicates}")
#     print(f"   • Final total records   : {after_dedup}")

#     return combined

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
        existing.sort_values(by=["Symbol", "Signal_Type", "Signal_date", "current_date"], inplace=True)
        existing = existing.drop_duplicates(subset=["Symbol", "Signal_Type", "Signal_date"], keep="last")

    # Identify new rows not in existing
    if existing.empty:
        new_records = new_df.copy()
    else:
        merged = new_df.merge(
            existing[["Symbol", "Signal_Type", "Signal_date"]],
            on=["Symbol", "Signal_Type", "Signal_date"],
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
