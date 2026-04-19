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
        WITH GET_SIGNALS AS (
            SELECT DISTINCT Symbol, DATE(Date) AS Signal_date, 
            'Accumulation' AS Signal_Type, 
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
            -- order by   Signal_date DESC, Signal_Rank ASC
        )
        SELECT A.* , B.UPDATE_DATE, B.Close AS current_price, b.Date as current_date
        FROM GET_SIGNALS AS A 
        LEFT JOIN STOCK_PRICES AS B
        ON upper(A.Symbol) = upper(B.Symbol)
        order by   Signal_date DESC, Signal_Rank ASC
    """
    df = pd.read_sql_query(signals_query, conn)
    print(f"✅ Total signals Present: {df.shape[0]}")
    return df

def merge_signals_with_prices(signals_df, stock_df):
    """Join signal data with stock prices to get post-signal price evolution."""
    
    # ✅ FIX: Flatten MultiIndex columns if yfinance returned them
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df = stock_df.copy()
        stock_df.columns = [col[0] if col[1] == '' else col[0] for col in stock_df.columns]
    
    format_stock_df = stock_df[["Symbol", "Date", "Close"]].copy()
    format_stock_df["Date"] = pd.to_datetime(format_stock_df["Date"])

    # ✅ FIX: Ensure Signal_Price is numeric before merge
    signals_df["Signal_Price"] = pd.to_numeric(signals_df["Signal_Price"], errors="coerce")

    merged = pd.merge(signals_df, format_stock_df, on="Symbol", how="inner")
    merged = merged[["Symbol", "Signal_date", "Signal_Type", "Signal_Price", "Signal_Rank", "Date", "Close"]]
    
    # ✅ FIX: Drop rows where Signal_Price is NaN before any calculations
    merged = merged.dropna(subset=["Signal_Price"])
    merged = merged[merged["Date"] >= pd.to_datetime(merged["Signal_date"])]
    merged = merged.sort_values(
        by=["Symbol", "Signal_Type", "Signal_date", "Date"],
        ascending=[True, True, False, False],
    )
    merged.rename(columns={"Close": "Price"}, inplace=True)

    print(f"✅ Total rows after merging signals with future prices: {merged.shape[0]}")
    return merged


def compute_returns(group):
    """Compute fixed returns + max return + drawdown + quality metrics for each horizon + full-period metrics."""

    group = group.sort_values("Date").reset_index(drop=True)

    # ensure numeric
    group["Price"] = pd.to_numeric(group["Price"], errors="coerce")
    group["Updated_Signal_Price"] = pd.to_numeric(group["Updated_Signal_Price"], errors="coerce")

    sig_date = group.at[0, "Signal_date"]
    sig_price = group.at[0, "Updated_Signal_Price"]

    out = {
        "Symbol": group.at[0, "Symbol"],
        "Signal_Type": group.at[0, "Signal_Type"],
        "Signal_date": sig_date,
        "Updated_Signal_Price": sig_price,
        "Signal_Rank": group.at[0, "Signal_Rank"],
    }

    for name, days in HORIZONS.items():

        # =========================
        # 1️⃣ FIXED RETURN
        # =========================
        target = sig_date + pd.Timedelta(days=days)
        sel = group[group["Date"] >= target]

        if not sel.empty:
            row = sel.iloc[0]
            p = row["Price"]
            d = row["Date"]

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

        # =========================
        # 2️⃣ WINDOW METRICS
        # =========================
        end_date = sig_date + pd.Timedelta(days=days)

        window = group[
            (group["Date"] > sig_date) &
            (group["Date"] <= end_date)
        ].copy()

        window = window[window["Price"].notna()]

        if not window.empty and pd.notna(sig_price) and sig_price != 0:

            window["ret"] = window["Price"] / sig_price - 1

            # MAX RETURN
            idx_max = window["ret"].idxmax()
            row_max = window.loc[idx_max]

            max_ret = row_max["ret"]
            max_price = row_max["Price"]
            max_date = row_max["Date"]

            # MAX DRAWDOWN
            idx_min = window["ret"].idxmin()
            row_min = window.loc[idx_min]

            min_ret = row_min["ret"]
            min_price = row_min["Price"]
            min_date = row_min["Date"]

            # time metrics
            time_to_peak = (max_date - sig_date).days
            time_to_dd = (min_date - sig_date).days

            # peak to end
            end_ret = window.iloc[-1]["ret"]
            peak_to_end = end_ret - max_ret

            # % days in profit
            pct_days_profit = (window["ret"] > 0).mean()

            # volatility
            vol = window["ret"].std()

            # store
            out[f"{name}_max"] = round(max_ret * 100, 2)
            out[f"{name}_max_price"] = max_price
            out[f"{name}_max_date"] = max_date

            out[f"{name}_dd"] = round(min_ret * 100, 2)
            out[f"{name}_dd_price"] = min_price
            out[f"{name}_dd_date"] = min_date

            out[f"{name}_time_to_peak"] = time_to_peak
            out[f"{name}_time_to_dd"] = time_to_dd

            out[f"{name}_peak_to_end"] = round(peak_to_end * 100, 2)
            out[f"{name}_pct_days_profit"] = round(pct_days_profit, 2)
            out[f"{name}_volatility"] = round(vol, 4)

        else:
            for suffix in [
                "_max", "_max_price", "_max_date",
                "_dd", "_dd_price", "_dd_date",
                "_time_to_peak", "_time_to_dd",
                "_peak_to_end", "_pct_days_profit", "_volatility"
            ]:
                out[f"{name}{suffix}"] = pd.NA

    # =========================
    # 3️⃣ FULL PERIOD METRICS (NEW)
    # =========================
    full_window = group[group["Date"] > sig_date].copy()
    full_window = full_window[full_window["Price"].notna()]

    if not full_window.empty and pd.notna(sig_price) and sig_price != 0:

        full_window["ret"] = full_window["Price"] / sig_price - 1

        # MAX RETURN
        idx_max = full_window["ret"].idxmax()
        row_max = full_window.loc[idx_max]

        out["ret_sinceSignal_max"] = round(row_max["ret"] * 100, 2)
        out["ret_sinceSignal_max_price"] = row_max["Price"]
        out["ret_sinceSignal_max_date"] = row_max["Date"]

        # MAX DRAWDOWN
        idx_min = full_window["ret"].idxmin()
        row_min = full_window.loc[idx_min]

        out["ret_sinceSignal_dd"] = round(row_min["ret"] * 100, 2)
        out["ret_sinceSignal_dd_price"] = row_min["Price"]
        out["ret_sinceSignal_dd_date"] = row_min["Date"]

    else:
        out["ret_sinceSignal_max"] = pd.NA
        out["ret_sinceSignal_max_price"] = pd.NA
        out["ret_sinceSignal_max_date"] = pd.NaT

        out["ret_sinceSignal_dd"] = pd.NA
        out["ret_sinceSignal_dd_price"] = pd.NA
        out["ret_sinceSignal_dd_date"] = pd.NaT

    # =========================
    # 4️⃣ CURRENT RETURN (keep this)
    # =========================
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
    """Apply return computation across all signal groups.

    # """

    #  DEBUG 18mar2026
    # print("DEBUG inside calculate_returns:", merged_df.columns.tolist())
  

    merged_df["Signal_date"] = pd.to_datetime(merged_df["Signal_date"])
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    
    # ✅ FIX: Ensure Signal_Price has no NaNs before groupby
    merged_df["Updated_Signal_Price"] = pd.to_numeric(merged_df["Updated_Signal_Price"], errors="coerce")

    # If Updated_Signal_Price is missing use Signal_Price instead for those records and then drop any remaining NaNs
    merged_df["Updated_Signal_Price"] = merged_df.apply(
        lambda row: row["Signal_Price"] if pd.isna(row["Updated_Signal_Price"]) else row["Updated_Signal_Price"],
        axis=1
    )
    merged_df = merged_df.dropna(subset=["Updated_Signal_Price"])

    returns_df = (
        merged_df.groupby(
            ["Symbol", "Signal_Type", "Signal_date", "Updated_Signal_Price", "Signal_Rank"],
            # as_index=False,
            sort=False,
            dropna=False,  # ✅ FIX: prevents KeyError on NaN keys
            group_keys=False   # ✅ ADD THIS
        )
        .apply(compute_returns)
        .reset_index(drop=True)
    )
    # print(returns_df.columns)
    print(f"✅ Total records after return calculation: {returns_df.shape[0]}")

    # Keep original Signal_Price for reference also in the final output, but use Updated_Signal_Price for all return calculations.
    #write a merge function to merge back the original Signal_Price from merged_df to returns_df based on Symbol, Signal_Type, Signal_date, Signal_Rank 


    returns_df = pd.merge(
        returns_df,
        merged_df[["Symbol", "Signal_Type", "Signal_date", "Signal_Rank", "Signal_Price"]].drop_duplicates(),
        on=["Symbol", "Signal_Type", "Signal_date", "Signal_Rank"],
        how="left"
    )

    #drop all other columns except Signal_Price after merge
    # col_order = ['Symbol', 'Signal_Type', 'Signal_date', 'Signal_Price','Updated_Signal_Price','Signal_Rank', 
    #              'ret_1w', 'ret_1w_price', 'ret_1w_date', 'ret_2w','ret_2w_price', 'ret_2w_date', 'ret_1m', 
    #              'ret_1m_price', 'ret_1m_date','ret_3m', 'ret_3m_price', 'ret_3m_date', 'ret_6m', 'ret_6m_price',
    #              'ret_6m_date', 'ret_1y', 'ret_1y_price', 'ret_1y_date','ret_sinceSignal', 'current_price', 'current_date' ]
    # returns_df = returns_df[col_order]

    # Convert numeric columns to ensure no text values
    ret_cols = [c for c in returns_df.columns if c.startswith("ret_") and not c.endswith("_price") and not c.endswith("_date")]
    for c in ret_cols:
        returns_df[c] = pd.to_numeric(returns_df[c], errors="coerce").round(2)

    price_cols = [c for c in returns_df.columns if c.endswith("_price")]
    for c in price_cols:
        returns_df[c] = pd.to_numeric(returns_df[c], errors="coerce")

    returns_df["Updated_Signal_Price"] = pd.to_numeric(returns_df["Updated_Signal_Price"], errors="coerce")

    # Calculate percentage returns based on *_price and Signal_Price
    for horizon in HORIZONS.keys():
        returns_df[f"{horizon}_perc"] = returns_df.apply(
            lambda row: round((row[f"{horizon}_price"] / row["Updated_Signal_Price"] - 1) * 100, 2)
            if pd.notna(row[f"{horizon}_price"]) and pd.notna(row["Updated_Signal_Price"]) and row["Updated_Signal_Price"] != 0
            else pd.NA,
            axis=1
        )

    # ret_sinceSignal_perc
    returns_df["ret_sinceSignal_perc"] = returns_df.apply(
        lambda row: round((row["current_price"] / row["Updated_Signal_Price"] - 1) * 100, 2)
        if pd.notna(row["current_price"]) and pd.notna(row["Updated_Signal_Price"]) and row["Updated_Signal_Price"] != 0
        else pd.NA,
        axis=1
    )

    # print(returns_df.columns)
    print(f"✅ Total records after return calculation: {returns_df.shape[0]}")

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
# Update Signal_Price for corporate actions
# ============================================================


def updated_signal_price(signals_df, stock_df):
    """Adjust Signal_Price for corporate actions- From stock_df get the price on signal date
    and call it Updated_Signal_Price. This will be used to calculate returns instead of original Signal_Price 
    """
    format_stock_df = stock_df[["Symbol", "Date", "Close"]].copy()
    format_stock_df["Date"] = pd.to_datetime(format_stock_df["Date"])

    # Ensure Signal_date is datetime
    signals_df["Signal_date"] = pd.to_datetime(signals_df["Signal_date"])

    merged = pd.merge(
        signals_df,
        format_stock_df.rename(columns={"Date": "Signal_date", "Close": "Updated_Signal_Price"}),
        on=["Symbol", "Signal_date"],
        how="left"
    )

    print(f"✅ Total signals after merging for updated Signal Price: {merged.shape[0]}")
    return merged


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

        # Step 1: Load signals from DB, adds current_date and current Price
        signals_df = load_signals_from_db(conn)
        if signals_df.empty:
            print("⚠️ No signals found in DB — skipping returns generation.")
            return True, pd.DataFrame()
        # print(f"load_signals_from_db ran :")
        # display(signals_df.tail(10))

        # Step 2: Merge signals with stock prices
        merged_df = merge_signals_with_prices(signals_df, stock_df)
        if merged_df.empty:
            print("⚠️ Merged DataFrame is empty — no price data matched signals.")
            return True, pd.DataFrame()

        # print(f"merge_signals_with_prices ran :")
        # display(merged_df.tail(10))

        # DEBUG - 18Mar2026
        # print("DEBUG merged_df columns:", merged_df.columns.tolist())

        # New function - Updated_Signal_Price - adjust for corporate actions .
        updated_signal_price_df = updated_signal_price(merged_df, stock_df)
        if updated_signal_price_df.empty: 
            print("⚠️ Merged DataFrame is empty after updated_signal_price — check for issues in price matching.")
            return True, pd.DataFrame() 
        
        # print(f"updated_signal_price ran :")
        # display(updated_signal_price_df.tail(10))

        # Step 3: Compute returns
        returns_df = calculate_returns(updated_signal_price_df)
        if returns_df.empty:
            print("⚠️ Returns DataFrame is empty after computation.")
            print(returns_df.head())
            return True, pd.DataFrame()

        # Instead of insert, replace the entire SIGNAL_RETURNS table every time. 
        # Step 4: Filter to only new records not already saved
        # new_records = get_new_records_only(conn, returns_df)

        # Step 5: Save to DB
        # if not new_records.empty:
        #     try:
        #         new_records.to_sql('SIGNAL_RETURNS', conn, if_exists='append', index=False)
        #         conn.commit()
        #         print("💾 SIGNAL_RETURNS table updated successfully.")
        #     except Exception as e:
        #         conn.rollback()
        #        print(f"❌ Error saving to SIGNAL_RETURNS: {e}")
        #        raise
        #else:
        #    print("ℹ️ No new records to save — SIGNAL_RETURNS already up to date.")

        #Save to DB - REPLACE ENTIRE TABLE
        if not returns_df.empty:
            try:
                df = returns_df.copy()
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime("%Y-%m-%d")
                    elif df[col].dtype == "object":
                        df[col] = df[col].apply(
                            lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x
                        )

                df = df.where(pd.notnull(df), None)
                df.to_sql('SIGNAL_RETURNS', conn, if_exists='replace', index=False)
                conn.commit()
                print("💾 SIGNAL_RETURNS table replaced successfully with new data.")
            except Exception as e:
                conn.rollback()
                print(f"❌ Error saving to SIGNAL_RETURNS: {e}")
                raise
        else:
            print("ℹ️ No Signals — SIGNAL_RETURNS is not replaced.")

        return True, df

    except Exception as e:
        print(f"❌ Error during signal returns generation: {e}")
        return False, None

    finally:
        if conn:
            conn.close()
            print("🔒 Database connection closed.")
