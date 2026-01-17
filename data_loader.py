"""
Market Oracle - Data Loader Module

Fetches stock data from Yahoo Finance and prepares it for analysis.
"""

from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_path: Optional path to save CSV
        
    Returns:
        DataFrame with OHLCV data
    """
    # TODO: Implement in Phase 1, Day 1-2
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data by handling missing values.
    
    Args:
        df: Raw DataFrame with OHLCV columns
        
    Returns:
        Cleaned DataFrame
    """
    # TODO: Implement in Phase 1, Day 2
    pass


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Compute log returns from price series.
    
    Formula: log_return = log(price_t / price_{t-1})
    
    Args:
        df: DataFrame with price data
        price_col: Column name for close price
        
    Returns:
        Series of log returns
    """
    # TODO: Implement in Phase 1, Day 2
    pass


if __name__ == "__main__":
    # Quick test
    print("Data loader module ready!")
