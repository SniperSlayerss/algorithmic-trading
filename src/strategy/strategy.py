from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path

"""
A Strategy should be able to...
- Backtest
  > Mark entry points from batch of data, return entry points
  > Backtest over csv of data (static)
...
"""


class Strategy(ABC):
    @classmethod
    def run_strategy(cls, financial_data: pd.DataFrame) -> np.ndarray:
        """

        Returns
        -------
        pd.DataFrame
            Contains entry points and exit points of strategy
        """
        pass

    @abstractmethod
    def is_entry(self, financial_data: pd.DataFrame) -> bool:
        """

        Returns
        -------
        bool
            Is the last point in financial_data considered an entry point
        """
        pass

    @staticmethod
    def fetch_data(financial_data: Path) -> pd.DataFrame:
        """
        CSV Format
        ----------
        timestamp, open, high, low, close,
        volume, close_time, quote_asset_volume,
        trades, taker_base_volume, taker_quote_volume, ignore
        """
        df = pd.read_csv(financial_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def run_backtest(financial_data: pd.DataFrame) -> None:
        pass
