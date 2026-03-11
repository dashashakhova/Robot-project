import unittest

import numpy as np
import pandas as pd

from core.market_context import MarketContextAnalyzer


def make_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="min")
    base = 100 + np.cumsum(np.random.normal(0, 0.05, rows))
    return pd.DataFrame(
        {
            "open": base + np.random.normal(0, 0.02, rows),
            "high": base + np.random.uniform(0.01, 0.15, rows),
            "low": base - np.random.uniform(0.01, 0.15, rows),
            "close": base,
            "volume": np.random.randint(100, 2000, rows),
        },
        index=idx,
    )


class TestMarketContext(unittest.TestCase):
    def test_analyze_returns_expected_sections(self):
        np.random.seed(1)
        analyzer = MarketContextAnalyzer()
        context = analyzer.analyze(make_ohlcv())

        self.assertIn("regime", context)
        self.assertIn("trend", context)
        self.assertIn("volatility", context)
        self.assertIn("support_resistance", context)
        self.assertIn("liquidity", context)

    def test_should_trade_contract(self):
        np.random.seed(2)
        analyzer = MarketContextAnalyzer()
        context = analyzer.analyze(make_ohlcv())
        result = analyzer.should_trade(context)

        self.assertIn("should_trade", result)
        self.assertIn("reasons", result)
        self.assertIsInstance(result["reasons"], list)


if __name__ == "__main__":
    unittest.main()
