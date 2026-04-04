# get the list of NSE Listed stocks from the database
import os
import pandas as pd
import logging
import yfinance as yf
import sqlite3
import datetime
import time

# ---------------------------
# Setup basic logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Database helper functions 
# ---------------------------

def get_company_symbols(db_path, table_name: str = "STOCK_DETAILS") -> pd.DataFrame:
    """
    Fetch company details (e.g., ticker symbols) from SQLite.
    Expects a database path.
    """
    print(db_path)
    """Create a connection to the SQLite database."""
    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        raise FileNotFoundError(f"Database not found at: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name} WHERE EXCHANGE != 'BSE' and UPDATE_DATE = (SELECT MAX(UPDATE_DATE) FROM STOCK_DETAILS)"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        logger.error(f"Error reading table {table_name}: {e}")
        raise
        
    # Add .NS in Symbol names to align with yfinance API format
    df['ticker'] = df['SYMBOL'] + '.NS'

    return df


# ----------------------------
# Yahoo Finance helper
# ----------------------------
# Download past 'raw_data_needed' year stock data 


def fetch_stock_data(tickers, raw_data_needed=365, interval="1d"):
    """
    Fetch stock data from Yahoo Finance for the given tickers,
    calculate EMAs and rolling volume averages.

    Args:
        tickers (list[str] or str): List of tickers or single ticker symbol.
        raw_data_needed (int): Number of days of history to fetch.
        interval (str): Data interval (default: '1d').

    Returns:
        pd.DataFrame: DataFrame containing price, EMAs, volume averages, etc.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    logger.info(f"📈 Fetching data for {len(tickers)} tickers...")

    start = pd.Timestamp.today() - datetime.timedelta(days=raw_data_needed)
    end = pd.Timestamp.today()

    logger.info(f"Fetching data from {start.date()} to {end.date()}")
    start_time = time.perf_counter()

    try:
        data = yf.download(tickers, start=start, end=end, threads=True, interval=interval)
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return pd.DataFrame()

    elapsed = time.perf_counter() - start_time
    logger.info(f"✅ Download completed in {elapsed:.2f} seconds")

    # Handle case: no data fetched
    if data.empty:
        logger.warning("No data returned by yfinance.")
        return pd.DataFrame()

    # Stack multi-index DataFrame into long format
    df = (
        data.stack()
        .reset_index()
        .rename(columns={"level_1": "Ticker"})
        .sort_values(["Ticker", "Date"])
    )

    # --- Technical indicators ---
    df["20ema"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["50ema"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["100ema"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    df["200ema"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=200, adjust=False).mean())
    df["30dv"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(window=30).mean())
    df["7dv"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(window=7).mean())

    # Clean up symbol names
    df["Symbol"] = df["Ticker"].str.replace(".NS", "", regex=False)

    df.reset_index(drop=True, inplace=True)
    logger.info(f"📊 Final dataset: {len(df):,} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


# Function to update STOCK_PRICES table in the database with latest stock prices for all stocks in STOCK_DETAILS table-
def update_current_stock_prices(db_path, stock_df_latest):
    """
    Update the STOCK_PRICES table in the database with the latest stock prices.

    Args:
        db_path (str): Path to the SQLite database.
        stock_df_latest (pd.DataFrame): DataFrame containing latest stock price data to be updated.
    """
    logger.info("🔄 Updating STOCK_PRICES table with latest stock prices...")

    try:
        conn = sqlite3.connect(db_path)
        #Read the STOCK_DETAILS table from the database
        stock_details = pd.read_sql("SELECT * FROM STOCK_DETAILS WHERE UPDATE_DATE = (SELECT MAX(UPDATE_DATE) FROM STOCK_DETAILS);", conn)
        
        #join the the latest stock prices. 
        joined_df = pd.merge(stock_details, stock_df_latest, left_on='SYMBOL', right_on='Symbol', how='inner')
        joined_df = joined_df[['ISIN_NUMBER', 'SYMBOL', 'EXCHANGE',  'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        joined_df['UPDATE_DATE'] = datetime.datetime.now().strftime('%Y-%m-%d')
        #REMOVE ROWS WHERE CLOSE IS NULL
        joined_df = joined_df[joined_df['Close'].notna()]
        joined_df.to_sql("STOCK_PRICES", conn, if_exists="replace", index=False)
        logger.info("✅ STOCK_PRICES table updated successfully.")
    except Exception as e:
        logger.error(f"Error updating STOCK_PRICES: {e}")
        raise
    finally:
        conn.close()

    logger.info(f"📊 STOCK_PRICES table updated for : {stock_df_latest['Date'].max().date()}")