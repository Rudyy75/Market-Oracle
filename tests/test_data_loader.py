"""
Market Oracle - Test Suite for Data Loader

Run with: pytest tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
import numpy as np

# TODO: Import from data_loader when implemented
# from data_loader import download_ticker_data, clean_data, compute_log_returns


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_download_returns_dataframe(self):
        """Test that download returns a valid DataFrame."""
        # TODO: Implement in Phase 1, Day 2
        pass
    
    def test_clean_handles_missing_values(self):
        """Test that NA values are handled correctly."""
        # TODO: Implement in Phase 1, Day 2
        pass
    
    def test_log_returns_calculation(self):
        """Test log return formula correctness."""
        # Sample data
        prices = pd.Series([100, 110, 105, 115])
        
        # Expected: log(110/100), log(105/110), log(115/105)
        expected = np.log(prices / prices.shift(1))
        
        # TODO: Compare with compute_log_returns output
        pass
    
    def test_log_returns_first_value_is_nan(self):
        """First log return should be NaN (no previous price)."""
        # TODO: Implement in Phase 1, Day 2
        pass
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty DataFrame."""
        # TODO: Implement in Phase 1, Day 2
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
