import pandas as pd
from utils.db import fetch_table


def build_features(db_path: str) -> pd.DataFrame:
    """
    Build ML-ready features from SIGNAL_RETURNS table
    """

    query = """
    SELECT *
    FROM SIGNAL_RETURNS
    """

    df = fetch_table(query, db_path)

    if df.empty:
        raise ValueError("No data found in SIGNAL_RETURNS")

    df = df.copy()

    # -------------------------------
    # Standardize column names
    # -------------------------------
    df.columns = [col.lower() for col in df.columns]

    # -------------------------------
    # Date handling
    # -------------------------------
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["current_date"] = pd.to_datetime(df["current_date"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    df["days_since_signal"] = (today - df["signal_date"]).dt.days

    # -------------------------------
    # Price-based features
    # -------------------------------
    df["price_vs_signal"] = (
        (df["current_price"] - df["signal_price"]) / df["signal_price"]
    )

    # -------------------------------
    # Return features (core signals)
    # -------------------------------
    return_cols = [
        "ret_1w_perc",
        "ret_2w_perc",
        "ret_1m_perc",
        "ret_3m_perc",
        "ret_6m_perc",
        "ret_1y_perc",
        "ret_sincesignal_perc",
    ]

    for col in return_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------
    # Signal type encoding
    # -------------------------------
    df["signal_type"] = df["signal_type"].str.lower()

    df["signal_type_encoded"] = df["signal_type"].map({
        "golden cross": 1,
        "accumulation": 0
    }).fillna(-1)

    # -------------------------------
    # Optional: interaction features
    # -------------------------------
    df["ret_1w_2w_combo"] = df["ret_1w_perc"] * df["ret_2w_perc"]
    df["momentum_acceleration"] = df["ret_1w_perc"] - df["ret_2w_perc"]

    return df