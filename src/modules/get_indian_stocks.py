import sqlite3
import pandas as pd
import os
from datetime import datetime


def get_indian_stocks(data_folder="../data/securities", db_path="../data/stocks.db", table_name: str = "STOCK_DETAILS_US") -> pd.DataFrame:

    # 1. Load NSE data
    nse_file = os.path.join(
        data_folder,  "EQUITY_L_20261101.csv"
    )

    df_NSE = pd.read_csv(nse_file)
    df_NSE.columns = df_NSE.columns.str.strip()
    df_NSE["EXCHANGE"] = "NSE"

    # 2. Load BSE data
    bse_t0_file = os.path.join(data_folder, "BSE_EQT0_20261101.csv")
    bse_t1_file = os.path.join(data_folder, "BSE_Equity_20261101.csv")

    df_BSE_t0 = pd.read_csv(bse_t0_file, index_col=False)
    df_BSE_t1 = pd.read_csv(bse_t1_file, index_col=False)

    df_BSE = pd.concat(
        [df_BSE_t0, df_BSE_t1],
        ignore_index=True
    )

    print(
        f"BSE T0: {df_BSE_t0.shape}, "
        f"BSE T1: {df_BSE_t1.shape}, "
        f"Combined: {df_BSE.shape}"
    )

    df_BSE["EXCHANGE"] = "BSE"
    df_BSE.columns = df_BSE.columns.str.strip()

    df_BSE.rename(
        columns={
            "ISIN No": "ISIN NUMBER",
            "Security Name": "NAME OF COMPANY",
            "Security Id": "SYMBOL"
        },
        inplace=True
    )

    # Remove duplicate ISINs
    df_BSE.drop_duplicates(
        subset=["ISIN NUMBER"],
        inplace=True
    )

    # Remove Mutual Funds
    df_BSE = df_BSE[
        ~df_BSE["Issuer Name"].str.contains(
            "Mutual Fund", case=False, na=False
        )
    ]

    # 3. Merge NSE and BSE
    df_merged = pd.merge(
        df_NSE,
        df_BSE,
        on="ISIN NUMBER",
        how="outer",
        suffixes=("_NSE", "_BSE"),
        indicator=True
    )

    df_merged["SYMBOL"] = (
        df_merged["SYMBOL_NSE"]
        .combine_first(df_merged["SYMBOL_BSE"])
    )

    df_merged["NAME OF COMPANY"] = (
        df_merged["NAME OF COMPANY_NSE"]
        .combine_first(df_merged["NAME OF COMPANY_BSE"])
    )

    df_merged["EXCHANGE"] = df_merged["_merge"].map({
        "both": "BOTH",
        "left_only": "NSE",
        "right_only": "BSE"
    })

    # 4. Keep required columns
    df_merged = df_merged[
        [
            "ISIN NUMBER",
            "SYMBOL",
            "NAME OF COMPANY",
            "EXCHANGE",
            "DATE OF LISTING"
        ]
    ]

    # 5. Remove duplicates and sort
    df_stocks = df_merged.drop_duplicates(
        subset=["ISIN NUMBER"]
    )

    df_stocks = df_stocks.sort_values(
        by="SYMBOL"
    )

    df_stocks.columns = [
        "ISIN_NUMBER",
        "SYMBOL",
        "COMPANY_NAME",
        "EXCHANGE",
        "LISTING_DATE"
    ]

    # 6. Add update date
    df_stocks["UPDATE_DATE"] = datetime.now().strftime("%Y-%m-%d")

    # 7. Save to SQLite database
    with sqlite3.connect(db_path) as conn:
        df_stocks.to_sql(
            "STOCK_DETAILS",
            conn,
            if_exists="append", #replace
            index=False
        )

    print(f"Total Indian stocks loaded: {len(df_stocks)}")
    print(f"Database updated: {db_path}")

    # return df_stocks
