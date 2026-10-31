import sqlite3
import pandas as pd
import re
import logging
import os

logger = logging.getLogger(__name__)


def get_us_stocks(db_path, table_name: str = "STOCK_DETAILS_US") -> pd.DataFrame:
    """
    Get list of NASDAQ listed company details and create a table in sqlite database.
    Expects a database path.
    """
    print(db_path)

    """Create a connection to the SQLite database."""

    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        raise FileNotFoundError(f"Database not found at: {db_path}")
     

    # Load your Nasdaq screener file. Source: https://www.nasdaq.com/market-activity/stocks/screener 
    df = pd.read_csv(r"D:\Work\Projects\Stock_Market\Stock_Market\data\nasdaq_screener_1793423118225.csv")

    # Clean text columns
    df["Name"] = df["Name"].fillna("").str.strip()
    df["Industry"] = df["Industry"].fillna("").str.strip()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()


    # --------------------------------------------------
    # 1. Get official ETF and test issue flags
    # --------------------------------------------------

    base_url = "https://www.nasdaqtrader.com/dynamic/symdir/"

    nasdaq = pd.read_csv(
        base_url + "nasdaqlisted.txt",
        sep="|",
        dtype=str
    )

    other = pd.read_csv(
        base_url + "otherlisted.txt",
        sep="|",
        dtype=str
    )

    # Remove footer rows
    nasdaq = nasdaq[
        ~nasdaq["Symbol"].str.contains("File Creation Time", na=False)
    ]

    other = other[
        ~other["ACT Symbol"].str.contains("File Creation Time", na=False)
    ]

    # Identify unwanted securities
    bad_nasdaq = nasdaq[
        (nasdaq["ETF"] == "Y") |
        (nasdaq["Test Issue"] == "Y") |
        (nasdaq["NextShares"] == "Y")
    ]["Symbol"].tolist()

    bad_other = other[
        (other["ETF"] == "Y") |
        (other["Test Issue"] == "Y")
    ]["ACT Symbol"].tolist()

    bad_symbols = set(bad_nasdaq + bad_other)


    # --------------------------------------------------
    # 2. Exclude unwanted security types by name
    # --------------------------------------------------

    exclude_patterns = [
        r"\bETF\b",
        r"\bETN\b",
        r"\bExchange[- ]Traded Funds?\b",
        r"\bExchange[- ]Traded Notes?\b",
        r"\bMutual Funds?\b",
        r"\bMoney Market Funds?\b",
        r"\bClosed[- ]End Funds?\b",
        r"\bWarrants?\b",
        r"\bRights?\b",
        r"\bUnits?\b",
        r"\bPreferred\b",
        r"\bDebentures?\b",
        r"\bBonds?\b",
        r"\bNotes?\b",
        r"\bSubscription Receipts?\b",
        r"\bWhen Issued\b",
        r"\bDepositary Shares?\b",
        r"\bCertificates? of Deposit\b"
    ]

    exclude_regex = re.compile(
        "|".join(exclude_patterns),
        flags=re.IGNORECASE
    )

    # Preserve American Depositary Shares (ADRs)
    def is_unwanted_name(name):
        name = re.sub(
            r"\bAmerican Depositary Shares?\b",
            "",
            name,
            flags=re.IGNORECASE
        )
        return bool(exclude_regex.search(name))


    name_filter = df["Name"].apply(is_unwanted_name)


    # --------------------------------------------------
    # 3. Exclude unwanted industries
    # --------------------------------------------------

    exclude_industries = [
        "Blank Checks",
        "Trusts Except Educational Religious and Charitable"
    ]

    industry_filter = df["Industry"].isin(exclude_industries)


    # --------------------------------------------------
    # 4. Official symbol filter
    # --------------------------------------------------

    symbol_filter = df["Symbol"].isin(bad_symbols)


    # --------------------------------------------------
    # 5. Remove Penny stocks (Price < $5) and market cap < $50 million
    # --------------------------------------------------
    prices = pd.to_numeric(
        df["Last Sale"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )
    penny_stock_filter = prices < 5.0

    # Remove securities with market cap < $50 million
    market_cap = pd.to_numeric(
        df["Market Cap"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )
    small_cap_filter = market_cap < 50000000

    # --------------------------------------------------
    # 6. Combine all filters
    # --------------------------------------------------

    remove_filter = (
        name_filter |
        industry_filter |
        symbol_filter |
        penny_stock_filter |
        small_cap_filter
    )

    #Create an Include column filter 
    df["Exclude"] = remove_filter


    df.drop(columns=["Last Sale", "Net Change" , "% Change", "Volume"], inplace=True)

    #MAKE COLUMN NAMES UPPER_SNAKE_CASE

    df.columns = df.columns.str.upper()
    df.rename(columns={"NAME": "COMPANY_NAME","MARKET CAP": "MARKET_CAP", "IPO YEAR": "IPO_YEAR"}, inplace=True) 

    df["EXCHANGE"] = "NASDAQ"

    #Convert IPO_YEAR to integer, replace NaN with 0
    df["IPO_YEAR"] = pd.to_numeric(df["IPO_YEAR"], errors="coerce").fillna(0).astype(int)

    #Add UPDATED_DATE column with current date
    df["UPDATE_DATE"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    df.head()

    print("Original securities:", len(df))
    print("Excluded securities:", len(df[df["EXCLUDE"] == True]))
    print("Remaining securities:", len(df[df["EXCLUDE"] == False]))


    # print("\nRemaining by industry:")
    # print(cleaned["Industry"].value_counts().head(20))

    # display.head()

    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False) 
        conn.close()
    except Exception as e:
        logger.error(f"Error reading table {table_name}: {e}")
        raise
    
    # Add this as a new table in stocks.db sqlite database. Use the following code to save the cleaned dataframe to a SQLite database:

    # import sqlite3
    # df.to_sql("STOCK_DETAILS_US", sqlite3.connect("stocks.db"), if_exists="replace", index=False) 
    # sqlite3.connect("stocks.db").close()
