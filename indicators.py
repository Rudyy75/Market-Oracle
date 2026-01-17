"""
Market Oracle - Technical Indicators Module

Computes technical indicators for feature engineering.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: RSI lookback period (default 14)
        
    Returns:
        Series of RSI values (0-100)
    """
    # TODO: Implement in Phase 1, Day 3
    pass


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD indicator.
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # TODO: Implement in Phase 1, Day 3
    pass


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute Simple Moving Average.
    
    Args:
        prices: Series of closing prices
        period: SMA lookback period
        
    Returns:
        Series of SMA values
    """
    # TODO: Implement in Phase 1, Day 3
    pass


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute rolling standard deviation of returns.
    
    Args:
        returns: Series of log returns
        window: Rolling window size
        
    Returns:
        Series of volatility values
    """
    # TODO: Implement in Phase 1, Day 3
    pass


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    # TODO: Implement in Phase 1, Day 3
    pass


if __name__ == "__main__":
    print("Indicators module ready!")
