import os
import sys
import pandas as pd

# Ensure the src package is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from accumulation_signal import detect_gradual_accumulation

def test_detect_returns_empty_without_error():
    df = pd.DataFrame({
        "Symbol": ["AAA"],
        "Date": ["2020-01-01"],
        "Close": [1],
        "High": [1],
        "Low": [1],
        "Volume": [1],
        "20ema": [2],
        "50ema": [3],
        "200ema": [4],
        "7dv": [0],
        "30dv": [0]
    })
    result = detect_gradual_accumulation(df, ad_window=5)
    assert result.empty


def test_detect_returns_empty_without_error():
    df = pd.DataFrame({
        'Symbol': ['AAA'],
        'Date': ['2020-01-01'],
        'Close': [1],
        'High': [1],
        'Low': [1],
        'Volume': [1],
        '20ema': [2],
        '50ema': [3],
        '200ema': [4],
        '7dv': [0],
        '30dv': [0]
    })
    result = detect_gradual_accumulation(df, ad_window=5)
    assert result.empty
