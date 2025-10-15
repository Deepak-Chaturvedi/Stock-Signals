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

def get_company_symbols(db_path, table_name: str = "COMPANY_DETAILS") -> pd.DataFrame:
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
        query = f"SELECT * FROM {table_name} WHERE EXCHANGE != 'BSE'"
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

