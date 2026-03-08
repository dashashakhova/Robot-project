"""
Простой бэктестер для быстрой проверки стратегий
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt


class SimpleBacktester:
    """
    Минималистичный бэктестер без лишних сложностей
    """

    def __init__(self, strategy, initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity = [initial_capital]

    def run(self, df: pd.DataFrame, show_progress: bool = True) -> Dict:
        """
        Запускает бэктестинг
        """
        print(f"\n📊 Бэктестинг {self.strategy.name} на {len(df)} свечах")

        position = None
        self.capital = self.initial_capital
        self.trades = []
        self.equity = [self.initial_capital]

        for i in range(100, len(df)):  # начинаем с 100 свечи для индикаторов
            current_data = df.iloc[:i]
            current_price = current_data['close'].iloc[-1]

            # Получаем сигнал
            signal = self.strategy.get_signal(current_data)

            # Управление позицией
            if position is None and signal['signal'] != 'HOLD':
                # Открываем позицию
                position = {
                    'type': signal['signal'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'entry_time': current_data.index[-1],
                    'entry_idx': i
                }

                if show_progress:
                    print(f"\n🟢 {signal['signal']} at {signal['entry_price']:.2f} | {', '.join(signal['reason'])}")

            elif position is not None:
                # Проверяем условия закрытия
                close_reason = None
                exit_price = current_price

                # Стоп-лосс
                if position['type'] == 'BUY' and current_price <= position['stop_loss']:
                    close_reason = 'stop_loss'
                elif position['type'] == 'SELL' and current_price >= position['stop_loss']:
                    close_reason = 'stop_loss'

                # Тейк-профит
                elif position['type'] == 'BUY' and current_price >= position['take_profit']:
                    close_reason = 'take_profit'
                elif position['type'] == 'SELL' and current_price <= position['take_profit']:
                    close_reason = 'take_profit'

                # Выход по времени (8 часов)
                hours_held = (current_data.index[-1] - position['entry_time']).total_seconds() / 3600
                if hours_held >= 8:
                    close_reason = 'timeout'

                if close_reason:
                    # Закрываем позицию
                    if position['type'] == 'BUY':
                        profit = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        profit = (position['entry_price'] - exit_price) / position['entry_price']

                    # Комиссия
                    profit -= 0.001  # 0.1%

                    self.capital *= (1 + profit)

                    self.trades.append({
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_time': position['entry_time'],
                        'exit_time': current_data.index[-1],
                        'profit': profit,
                        'reason': close_reason
                    })

                    if show_progress:
                        print(
                            f"🔴 {position['type']} closed at {exit_price:.2f} | profit: {profit:.2%} | {close_reason}")

                    position = None

            # Обновляем кривую капитала
            if position:
                if position['type'] == 'BUY':
                    unrealized = (current_price - position['entry_price']) / position['entry_price']
                else:
                    unrealized = (position['entry_price'] - current_price) / position['entry_price']
                self.equity.append(self.capital * (1 + unrealized))
            else:
                self.equity.append(self.capital)

        # Закрываем последнюю позицию
        if position:
            exit_price = df['close'].iloc[-1]
            if position['type'] == 'BUY':
                profit = (exit_price - position['entry_price']) / position['entry_price']
            else:
                profit = (position['entry_price'] - exit_price) / position['entry_price']

            profit -= 0.001
            self.capital *= (1 + profit)

            self.trades.append({
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'profit': profit,
                'reason': 'end_of_test'
            })

        # Рассчитываем метрики
        results = self._calculate_metrics()

        print(f"\n✅ Результаты:")
        print(f"   Сделок: {results['total_trades']}")
        print(f"   Прибыльных: {results['winning_trades']} ({results['win_rate']:.1%})")
        print(f"   Доходность: {results['total_return']:.2%}")
        print(f"   Финальный капитал: {results['final_capital']:,.0f}")

        return results

    def _calculate_metrics(self) -> Dict:
        """Рассчитывает базовые метрики"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'final_capital': self.initial_capital
            }

        profits = [t['profit'] for t in self.trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades),
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'final_capital': self.capital,
            'avg_profit': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'trades': self.trades[-20:]  # последние 20 сделок
        }

    def plot_equity(self):
        """Рисует кривую капитала"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity, label='Equity', color='blue')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial')
        plt.title(f'{self.strategy.name} - Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()