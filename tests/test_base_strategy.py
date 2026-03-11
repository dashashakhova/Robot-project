import unittest

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


def make_ohlcv(rows: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="min")
    base = 100 + np.cumsum(np.random.normal(0, 0.05, rows))
    return pd.DataFrame(
        {
            "open": base + np.random.normal(0, 0.01, rows),
            "high": base + np.random.uniform(0.01, 0.12, rows),
            "low": base - np.random.uniform(0.01, 0.12, rows),
            "close": base,
            "volume": np.random.randint(200, 1500, rows),
        },
        index=idx,
    )


class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Dummy", "VKCO")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signal(self, df: pd.DataFrame):
        price = float(df["close"].iloc[-1])
        return {
            "action": "BUY",
            "confidence": 0.9,
            "reasons": ["raw action format"],
            "entry_price": price,
            "stop_loss": price * 0.995,
            "take_profit": price * 1.01,
        }

    def get_params(self):
        return {}

    def set_params(self, params):
        return None


class TestBaseStrategy(unittest.TestCase):
    def test_post_processing_normalizes_signal(self):
        np.random.seed(3)
        strategy = DummyStrategy()
        signal = strategy.get_signal(make_ohlcv())

        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})
        self.assertEqual(signal["signal"], signal["action"])
        self.assertIsInstance(signal["reason"], list)
        self.assertIn("market_context", signal)
        self.assertIn("instrument", signal)
        self.assertIn("strategy", signal)


if __name__ == "__main__":
    unittest.main()
