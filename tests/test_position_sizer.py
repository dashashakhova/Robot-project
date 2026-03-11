import unittest

from risk.position_sizer import PositionSizeManager


class TestPositionSizer(unittest.TestCase):
    def test_position_value_respects_max_position_pct(self):
        manager = PositionSizeManager(capital=100_000)
        result = manager.calculate_position_size(
            entry_price=100.0,
            stop_loss_price=99.0,
            direction="LONG",
        )

        self.assertTrue(result["can_trade"])
        self.assertLessEqual(result["position_value"], 100_000 * manager.max_position_pct + 1e-6)

    def test_daily_loss_uses_downside_only(self):
        manager = PositionSizeManager(capital=100_000)
        manager.update_trade_result({"pnl": -4_000})
        manager.update_trade_result({"pnl": 10_000})

        # Профит не должен уменьшать накопленный downside
        self.assertEqual(manager.daily_loss, 4_000)

    def test_daily_loss_limit_blocks_new_trade(self):
        manager = PositionSizeManager(capital=100_000)
        manager.update_trade_result({"pnl": -6_000})  # больше 5% лимита по умолчанию

        result = manager.calculate_position_size(
            entry_price=100.0,
            stop_loss_price=99.0,
            direction="LONG",
        )
        self.assertFalse(result["can_trade"])
        self.assertEqual(result["reason"], "Daily loss limit reached")


if __name__ == "__main__":
    unittest.main()
