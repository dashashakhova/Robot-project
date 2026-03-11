"""
Простой бэктестер для быстрой проверки стратегий.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import config


class SimpleBacktester:
    """
    Минималистичный бэктестер с более реалистичным исполнением:
    - вход на следующей свече по open;
    - проверка стопа/тейка по high/low внутри свечи;
    - проскальзывание и комиссия на входе и выходе.
    """

    def __init__(self, strategy, initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity = [initial_capital]

        self.commission = config.get("trading", "commission", default=0.001)
        self.slippage = config.get("trading", "slippage", default=0.0005)
        self.max_hold_hours = 8

    def run(self, df: pd.DataFrame, show_progress: bool = True) -> Dict:
        """Запускает бэктестинг."""
        print(f"\n📊 Бэктестинг {self.strategy.name} на {len(df)} свечах")

        if len(df) < 120:
            print("⚠️ Недостаточно данных для бэктеста")
            return self._calculate_metrics()

        position: Optional[Dict] = None
        self.capital = self.initial_capital
        self.trades = []
        self.equity = [self.initial_capital]

        # Сигнал считаем по закрытым данным до i, исполняем на open свечи i
        for i in range(100, len(df)):
            signal_data = df.iloc[:i]
            bar = df.iloc[i]
            bar_time = df.index[i]

            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # 1) Сначала управляем уже открытой позицией на текущей свече
            if position is not None and i > position["entry_idx"]:
                close_reason = None
                exit_price = None

                if position["type"] == "BUY":
                    # Консервативно: если в свече задеты и стоп, и тейк, считаем что сработал стоп
                    if bar_low <= position["stop_loss"]:
                        close_reason = "stop_loss"
                        exit_price = self._apply_exit_slippage(position["stop_loss"], position["type"])
                    elif bar_high >= position["take_profit"]:
                        close_reason = "take_profit"
                        exit_price = self._apply_exit_slippage(position["take_profit"], position["type"])
                else:  # SELL
                    if bar_high >= position["stop_loss"]:
                        close_reason = "stop_loss"
                        exit_price = self._apply_exit_slippage(position["stop_loss"], position["type"])
                    elif bar_low <= position["take_profit"]:
                        close_reason = "take_profit"
                        exit_price = self._apply_exit_slippage(position["take_profit"], position["type"])

                hours_held = (bar_time - position["entry_time"]).total_seconds() / 3600.0
                if close_reason is None and hours_held >= self.max_hold_hours:
                    close_reason = "timeout"
                    exit_price = self._apply_exit_slippage(bar_close, position["type"])

                if close_reason:
                    self._close_position(
                        position=position,
                        exit_price=exit_price,
                        exit_time=bar_time,
                        close_reason=close_reason,
                        show_progress=show_progress,
                    )
                    position = None

            # 2) Если позиции нет, можно открыть новую (по open текущей свечи)
            if position is None:
                signal = self.strategy.get_signal(signal_data)
                signal_type = signal.get("signal", "HOLD")

                if signal_type in {"BUY", "SELL"}:
                    stop_loss = signal.get("stop_loss")
                    take_profit = signal.get("take_profit")

                    if stop_loss is None or take_profit is None:
                        continue

                    entry_price = self._apply_entry_slippage(bar_open, signal_type)
                    if not self._is_valid_entry(signal_type, entry_price, float(stop_loss), float(take_profit)):
                        continue

                    position = {
                        "type": signal_type,
                        "entry_price": entry_price,
                        "stop_loss": float(stop_loss),
                        "take_profit": float(take_profit),
                        "entry_time": bar_time,
                        "entry_idx": i,
                        "signal_time": signal_data.index[-1] if len(signal_data) > 0 else bar_time,
                    }

                    if show_progress:
                        reasons = signal.get("reason", signal.get("reasons", []))
                        if isinstance(reasons, str):
                            reasons = [reasons]
                        reason_text = ", ".join(reasons) if reasons else "no reason"
                        print(f"\n🟢 {signal_type} at {entry_price:.2f} | {reason_text}")

            # 3) Обновляем кривую капитала
            if position:
                mark_price = self._apply_exit_slippage(bar_close, position["type"])
                if position["type"] == "BUY":
                    gross_unrealized = (mark_price - position["entry_price"]) / position["entry_price"]
                else:
                    gross_unrealized = (position["entry_price"] - mark_price) / position["entry_price"]

                # Учитываем round-trip комиссию в оценке open PnL
                net_unrealized = gross_unrealized - (2 * self.commission)
                self.equity.append(self.capital * (1 + net_unrealized))
            else:
                self.equity.append(self.capital)

        # Закрываем последнюю позицию по последнему close
        if position:
            exit_price = self._apply_exit_slippage(float(df["close"].iloc[-1]), position["type"])
            self._close_position(
                position=position,
                exit_price=exit_price,
                exit_time=df.index[-1],
                close_reason="end_of_test",
                show_progress=show_progress,
            )

        results = self._calculate_metrics()

        print(f"\n✅ Результаты:")
        print(f"   Сделок: {results['total_trades']}")
        print(f"   Прибыльных: {results['winning_trades']} ({results['win_rate']:.1%})")
        print(f"   Доходность: {results['total_return']:.2%}")
        print(f"   Финальный капитал: {results['final_capital']:,.0f}")

        return results

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_time,
        close_reason: str,
        show_progress: bool = True,
    ) -> None:
        if position["type"] == "BUY":
            gross_return = (exit_price - position["entry_price"]) / position["entry_price"]
            direction = "LONG"
        else:
            gross_return = (position["entry_price"] - exit_price) / position["entry_price"]
            direction = "SHORT"

        # Комиссия берется на входе и на выходе
        net_return = gross_return - (2 * self.commission)

        capital_before = self.capital
        self.capital *= (1 + net_return)
        pnl = self.capital - capital_before
        holding_period = (exit_time - position["entry_time"]).total_seconds() / 3600.0

        trade = {
            # Новый формат для monitoring/performance_tracker
            "instrument": getattr(self.strategy, "instrument", "UNKNOWN"),
            "direction": direction,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "pnl": pnl,
            "pnl_pct": net_return,
            "holding_period": holding_period,
            "reason": close_reason,
            # Обратная совместимость
            "type": position["type"],
            "profit": net_return,
            "gross_return": gross_return,
        }
        self.trades.append(trade)

        if show_progress:
            print(
                f"🔴 {position['type']} closed at {exit_price:.2f} | "
                f"profit: {net_return:.2%} | {close_reason}"
            )

    def _apply_entry_slippage(self, price: float, side: str) -> float:
        if side == "BUY":
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def _apply_exit_slippage(self, price: float, side: str) -> float:
        # BUY позицию закрываем продажей (цена хуже вниз), SELL закрываем покупкой (цена хуже вверх)
        if side == "BUY":
            return price * (1 - self.slippage)
        return price * (1 + self.slippage)

    @staticmethod
    def _is_valid_entry(side: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        if side == "BUY":
            return stop_loss < entry_price < take_profit
        if side == "SELL":
            return take_profit < entry_price < stop_loss
        return False

    def _calculate_metrics(self) -> Dict:
        """Рассчитывает базовые метрики."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "final_capital": self.initial_capital,
            }

        profits = [t["pnl_pct"] for t in self.trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.trades),
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "final_capital": self.capital,
            "avg_profit": np.mean(profits) if profits else 0,
            "max_profit": max(profits) if profits else 0,
            "max_loss": min(profits) if profits else 0,
            "trades": self.trades[-20:],  # последние 20 сделок
        }

    def plot_equity(self):
        """Рисует кривую капитала."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity, label="Equity", color="blue")
        plt.axhline(y=self.initial_capital, color="gray", linestyle="--", label="Initial")
        plt.title(f"{self.strategy.name} - Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
