import sqlite3
import pandas as pd


def get_connection(db_path: str):
    """
    Create SQLite connection
    """
    return sqlite3.connect(db_path)


def fetch_table(query: str, db_path: str) -> pd.DataFrame:
    """
    Execute SQL query and return DataFrame
    """

    conn = None
    try:
        conn = get_connection(db_path)
        df = pd.read_sql(query, conn)
        return df

    except Exception as e:
        raise RuntimeError(f"Error fetching data from DB: {e}")

    finally:
        if conn:
            conn.close()