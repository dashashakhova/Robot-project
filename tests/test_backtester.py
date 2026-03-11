import unittest

import numpy as np
import pandas as pd

from core.simple_backtester import SimpleBacktester
from strategies.base_strategy import BaseStrategy


def make_ohlcv(rows: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="min")
    trend = np.linspace(0, 3.5, rows)
    noise = np.random.normal(0, 0.02, rows)
    close = 100 + trend + noise
    return pd.DataFrame(
        {
            "open": close + np.random.normal(0, 0.01, rows),
            "high": close + np.random.uniform(0.03, 0.08, rows),
            "low": close - np.random.uniform(0.03, 0.08, rows),
            "close": close,
            "volume": np.random.randint(100, 1000, rows),
        },
        index=idx,
    )


class AlwaysBuyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("AlwaysBuy", "VKCO")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signal(self, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        return {
            "signal": "BUY",
            "confidence": 0.95,
            "entry_price": price,
            "stop_loss": price * 0.995,
            "take_profit": price * 1.005,
            "reason": ["test entry"],
        }

    def get_params(self):
        return {}

    def set_params(self, params):
        return None


class TestBacktester(unittest.TestCase):
    def test_trade_payload_contains_tracker_fields(self):
        np.random.seed(4)
        strategy = AlwaysBuyStrategy()
        backtester = SimpleBacktester(strategy, initial_capital=100_000)
        results = backtester.run(make_ohlcv(), show_progress=False)

        self.assertGreaterEqual(results["total_trades"], 1)
        last_trade = backtester.trades[-1]
        self.assertIn("instrument", last_trade)
        self.assertIn("direction", last_trade)
        self.assertIn("pnl", last_trade)
        self.assertIn("pnl_pct", last_trade)
        self.assertIn("profit", last_trade)
        self.assertIn(last_trade["direction"], {"LONG", "SHORT"})


if __name__ == "__main__":
    unittest.main()
