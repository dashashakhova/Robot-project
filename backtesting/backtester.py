"""
Модуль для продвинутого бэктестинга с реалистичными условиями
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

from core.config import config
from risk.position_sizer import PositionSizeManager
from risk.stop_loss_manager import StopLossManager, StopLossType
from risk.portfolio_risk import PortfolioRiskManager
from strategies.base_strategy import BaseStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Продвинутый бэктестер с реалистичными условиями торговли
    """

    def __init__(self, strategy: BaseStrategy, initial_capital: float = 1000000):
        """
        Args:
            strategy: торговая стратегия
            initial_capital: начальный капитал
        """
        self.strategy = strategy
        self.initial_capital = initial_capital

        # Компоненты риск-менеджмента
        self.position_manager = PositionSizeManager(initial_capital)
        self.stop_manager = StopLossManager(strategy.instrument)
        self.portfolio_manager = PortfolioRiskManager(initial_capital)

        # Параметры симуляции
        self.commission = config.get('trading', 'commission', default=0.001)
        self.slippage = config.get('trading', 'slippage', default=0.0005)
        self.min_confidence = config.get('strategy', 'min_confidence', default=0.6)

        # Результаты
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.signals_generated = 0
        self.signals_executed = 0

        # Для симуляции временных задержек
        self.latency_seconds = 2  # Задержка между сигналом и исполнением

    def run(self, df: pd.DataFrame, start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None, show_progress: bool = True) -> Dict:
        """
        Запускает бэктестинг

        Args:
            df: DataFrame с данными
            start_date: начальная дата
            end_date: конечная дата
            show_progress: показывать прогресс

        Returns:
            Словарь с результатами
        """
        # Фильтруем данные по датам
        test_df = df.copy()
        if start_date:
            test_df = test_df[test_df.index >= start_date]
        if end_date:
            test_df = test_df[test_df.index <= end_date]

        if len(test_df) == 0:
            logger.error("❌ Нет данных для тестирования")
            return {}

        logger.info(f"\n📊 Запуск бэктестинга для {self.strategy.instrument}")
        logger.info(f"   Период: {test_df.index[0]} - {test_df.index[-1]}")
        logger.info(f"   Свечей: {len(test_df)}")
        logger.info(f"   Начальный капитал: {self.initial_capital:,.0f} ₽")

        # Сбрасываем состояние
        self._reset()

        # Основной цикл тестирования
        for i in range(len(test_df) - max(self.strategy.forecast_hours)):
            current_time = test_df.index[i]
            current_data = test_df.iloc[:i + 1]  # Все данные до текущего момента

            # Получаем сигнал от стратегии
            signal = self.strategy.get_signal(current_data)

            if signal['action'] != 'HOLD':
                self.signals_generated += 1

            # Симулируем временную задержку
            execution_idx = min(i + 1, len(test_df) - 1)
            execution_price = self._simulate_execution_price(
                test_df, i, execution_idx, signal
            )

            # Проверяем существующие позиции
            self._check_positions(test_df.iloc[execution_idx])

            # Открываем новую позицию если есть сигнал
            if signal['action'] != 'HOLD' and signal['confidence'] >= self.min_confidence:
                self._execute_signal(signal, execution_price, current_time)

            # Обновляем кривую капитала
            self._update_equity_curve(current_time, test_df.iloc[execution_idx])

            if show_progress and i % 1000 == 0:
                logger.info(f"   Прогресс: {i}/{len(test_df)}")

        # Закрываем все позиции в конце
        self._close_all_positions(test_df.iloc[-1])

        # Рассчитываем метрики
        results = self._calculate_metrics()

        logger.info(f"\n✅ Бэктестинг завершен")
        logger.info(f"   Всего сделок: {results['total_trades']}")
        logger.info(f"   Доходность: {results['total_return']:.2%}")
        logger.info(f"   Финальный капитал: {results['final_capital']:,.0f} ₽")

        return results

    def _reset(self):
        """Сбрасывает состояние бэктестера"""
        self.trades = []
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.drawdown_curve = []
        self.signals_generated = 0
        self.signals_executed = 0
        self.position_manager = PositionSizeManager(self.initial_capital)
        self.portfolio_manager = PortfolioRiskManager(self.initial_capital)

    def _simulate_execution_price(self, df: pd.DataFrame, signal_idx: int,
                                  execution_idx: int, signal: Dict) -> float:
        """
        Симулирует цену исполнения с учетом задержки и проскальзывания

        Args:
            df: DataFrame с данными
            signal_idx: индекс свечи с сигналом
            execution_idx: индекс свечи исполнения
            signal: сигнал

        Returns:
            Цена исполнения
        """
        if signal_idx == execution_idx:
            # Мгновенное исполнение
            base_price = df.iloc[signal_idx]['close']
        else:
            # Исполнение на следующей свече
            base_price = df.iloc[execution_idx]['open']

        # Добавляем проскальзывание
        if signal['action'] == 'BUY':
            execution_price = base_price * (1 + self.slippage)
        else:
            execution_price = base_price * (1 - self.slippage)

        return execution_price

    def _execute_signal(self, signal: Dict, execution_price: float, signal_time: datetime):
        """
        Исполняет торговый сигнал

        Args:
            signal: сигнал
            execution_price: цена исполнения
            signal_time: время сигнала
        """
        # Конвертируем действие в направление
        direction = 'LONG' if signal['action'] == 'BUY' else 'SHORT'

        # Получаем ATR для расчета стопа
        atr = signal.get('market_context', {}).get('volatility', {}).get('current_atr_pct', 1.0)
        atr_value = execution_price * (atr / 100)

        # Получаем уровни поддержки/сопротивления
        support_levels = signal.get('market_context', {}).get('support_resistance', {}).get('support_levels', [])
        resistance_levels = signal.get('market_context', {}).get('support_resistance', {}).get('resistance_levels', [])

        # Создаем стоп-ордер
        stop_order = self.stop_manager.create_stop_order(
            trade_id=f"trade_{len(self.trades)}",
            entry_price=execution_price,
            direction=direction,
            atr=atr_value,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )

        # Рассчитываем размер позиции
        position_size = self.position_manager.calculate_position_size(
            entry_price=execution_price,
            stop_loss_price=stop_order['stop_loss'],
            direction=direction,
            atr=atr_value
        )

        if not position_size['can_trade']:
            logger.debug(f"   ⚠️ Сигнал отклонен: {position_size['reason']}")
            return

        # Корректируем с учетом волатильности
        position_size = self.position_manager.adjust_for_volatility(position_size, atr)

        # Добавляем позицию в портфель
        position_info = {
            'direction': direction,
            'entry_price': execution_price,
            'position_size': position_size['position_size'],
            'position_value': position_size['position_value'],
            'stop_loss': stop_order['stop_loss'],
            'take_profit': stop_order['take_profit'],
            'risk_amount': position_size['risk_amount'],
            'signal_confidence': signal['confidence'],
            'signal_time': signal_time
        }

        if self.portfolio_manager.add_position(self.strategy.instrument, position_info):
            self.signals_executed += 1
            logger.info(f"   🟢 {direction} {position_size['position_size']} @ {execution_price:.2f} "
                        f"(риск: {position_size['risk_pct']:.2%})")

    def _check_positions(self, current_candle: pd.Series):
        """
        Проверяет открытые позиции на срабатывание стопов

        Args:
            current_candle: текущая свеча
        """
        current_price = current_candle['close']

        for instrument in list(self.portfolio_manager.positions.keys()):
            position = self.portfolio_manager.positions[instrument]

            # Проверяем стопы
            stop_check = self.stop_manager.check_stop_trigger(
                f"trade_{len(self.trades)}",  # В реальном проекте нужен правильный trade_id
                current_price
            )

            if stop_check['triggered']:
                # Закрываем позицию
                trade_result = self.portfolio_manager.close_position(
                    instrument,
                    stop_check['price'],
                    stop_check['reason']
                )

                if trade_result:
                    self.trades.append(trade_result)
                    self.position_manager.update_trade_result(trade_result)

                    logger.info(f"   🔴 Закрытие {trade_result['direction']} "
                                f"прибыль: {trade_result['pnl_pct']:.2%} ({stop_check['reason']})")

            # Обновляем трейлинг стопы
            else:
                # Находим максимум/минимум для трейлинга
                # В реальном проекте нужно передавать историю цен
                pass

    def _close_all_positions(self, last_candle: pd.Series):
        """
        Закрывает все открытые позиции в конце теста

        Args:
            last_candle: последняя свеча
        """
        current_price = last_candle['close']

        for instrument in list(self.portfolio_manager.positions.keys()):
            trade_result = self.portfolio_manager.close_position(
                instrument,
                current_price,
                "End of test"
            )

            if trade_result:
                self.trades.append(trade_result)
                logger.info(f"   🔴 Закрытие {trade_result['direction']} "
                            f"прибыль: {trade_result['pnl_pct']:.2%} (End of test)")

    def _update_equity_curve(self, current_time: datetime, current_candle: pd.Series):
        """
        Обновляет кривую капитала

        Args:
            current_time: текущее время
            current_candle: текущая свеча
        """
        # Обновляем стоимость позиций
        market_data = {self.strategy.instrument: current_candle['close']}
        self.portfolio_manager.update_positions(market_data)

        # Получаем статус портфеля
        status = self.portfolio_manager.get_portfolio_status()

        # Сохраняем точку кривой капитала
        self.equity_curve.append((current_time, status['current_capital']))
        self.drawdown_curve.append((current_time, status['current_drawdown']))

    def _calculate_metrics(self) -> Dict:
        """
        Рассчитывает расширенные метрики эффективности

        Returns:
            Словарь с метриками
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'final_capital': self.initial_capital
            }

        # Основные метрики из портфеля
        portfolio_status = self.portfolio_manager.get_portfolio_status()

        # Анализ сделок
        trades_df = pd.DataFrame(self.trades)

        # Прибыльные и убыточные сделки
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        # Временной анализ
        trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600

        # Анализ по часам
        trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_stats = trades_df.groupby('entry_hour').agg({
            'pnl': ['mean', 'count', 'sum']
        }).round(4)

        # Метрики
        results = {
            # Общая статистика
            'total_trades': len(self.trades),
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'execution_rate': self.signals_executed / self.signals_generated if self.signals_generated > 0 else 0,

            # Капитал и доходность
            'initial_capital': self.initial_capital,
            'final_capital': portfolio_status['current_capital'],
            'total_return': portfolio_status['total_return'],
            'total_pnl': portfolio_status['current_capital'] - self.initial_capital,

            # Просадки
            'max_drawdown': portfolio_status['max_drawdown'],
            'avg_drawdown': np.mean([dd for _, dd in self.drawdown_curve]) if self.drawdown_curve else 0,
            'drawdown_days': len([dd for _, dd in self.drawdown_curve if dd > 0.05]),  # дни с просадкой >5%

            # Статистика по сделкам
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,

            'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['pnl_pct'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl_pct'].min() if len(losing_trades) > 0 else 0,

            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
            if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'),

            'avg_holding_hours': trades_df['holding_period'].mean(),
            'median_holding_hours': trades_df['holding_period'].median(),

            # Риск-метрики
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'calmar_ratio': portfolio_status['total_return'] / portfolio_status['max_drawdown']
            if portfolio_status['max_drawdown'] > 0 else 0,
            'var_95': portfolio_status['var_95'],

            # Детальная статистика
            'hourly_stats': hourly_stats.to_dict(),
            'trades': self.trades[-100:],  # последние 100 сделок
        }

        return results

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Рассчитывает коэффициент Шарпа"""
        if len(self.equity_curve) < 2:
            return 0

        returns = pd.Series([v for _, v in self.equity_curve]).pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0

        excess_returns = returns.mean() - risk_free_rate / (252 * 24)  # часовая безрисковая ставка
        return excess_returns / returns.std() * np.sqrt(252 * 24)

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Рассчитывает коэффициент Сортино (учитывает только отрицательную волатильность)"""
        if len(self.equity_curve) < 2:
            return 0

        returns = pd.Series([v for _, v in self.equity_curve]).pct_change().dropna()
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0 if returns.mean() <= 0 else float('inf')

        excess_returns = returns.mean() - risk_free_rate / (252 * 24)
        return excess_returns / negative_returns.std() * np.sqrt(252 * 24)

    def plot_results(self, save_path: Optional[str] = None):
        """
        Визуализирует результаты бэктестинга

        Args:
            save_path: путь для сохранения графика
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Результаты бэктестинга {self.strategy.instrument}', fontsize=16)

        # 1. Кривая капитала
        ax1 = axes[0, 0]
        times = [t for t, _ in self.equity_curve]
        equity = [v for _, v in self.equity_curve]
        ax1.plot(times, equity, label='Equity', color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Кривая капитала')
        ax1.set_xlabel('Дата')
        ax1.set_ylabel('Капитал')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Просадка
        ax2 = axes[0, 1]
        dd_times = [t for t, _ in self.drawdown_curve]
        dd_values = [v for _, v in self.drawdown_curve]
        ax2.fill_between(dd_times, dd_values, 0, color='red', alpha=0.3)
        ax2.set_title('Просадка')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Просадка %')
        ax2.grid(True, alpha=0.3)

        # 3. Распределение прибыли по сделкам
        ax3 = axes[1, 0]
        if self.trades:
            pnls = [t['pnl_pct'] * 100 for t in self.trades]
            ax3.hist(pnls, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--')
            ax3.set_title('Распределение прибыли по сделкам')
            ax3.set_xlabel('Прибыль %')
            ax3.set_ylabel('Частота')
            ax3.grid(True, alpha=0.3)

        # 4. Прибыль по часам
        ax4 = axes[1, 1]
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
            hourly_pnl = trades_df.groupby('hour')['pnl_pct'].mean() * 100
            ax4.bar(hourly_pnl.index, hourly_pnl.values, color='purple', alpha=0.7)
            ax4.set_title('Средняя прибыль по часам входа')
            ax4.set_xlabel('Час')
            ax4.set_ylabel('Средняя прибыль %')
            ax4.grid(True, alpha=0.3)

        # 5. Накопленная прибыль
        ax5 = axes[2, 0]
        if self.trades:
            cumulative_pnl = np.cumsum([t['pnl'] for t in self.trades])
            ax5.plot(range(len(cumulative_pnl)), cumulative_pnl, color='green')
            ax5.set_title('Накопленная прибыль')
            ax5.set_xlabel('Номер сделки')
            ax5.set_ylabel('Прибыль')
            ax5.grid(True, alpha=0.3)

        # 6. Месячная доходность
        ax6 = axes[2, 1]
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
            equity_df.set_index('date', inplace=True)
            monthly_returns = equity_df['equity'].resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.dropna()
            ax6.bar(range(len(monthly_returns)), monthly_returns.values, color='orange', alpha=0.7)
            ax6.set_title('Месячная доходность')
            ax6.set_xlabel('Месяц')
            ax6.set_ylabel('Доходность %')
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 График сохранен в {save_path}")

        plt.show()

    def save_results(self, filename: Optional[str] = None):
        """
        Сохраняет результаты в JSON

        Args:
            filename: имя файла
        """
        if filename is None:
            filename = f"backtest_{self.strategy.instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = self._calculate_metrics()

        # Конвертируем datetime в строки
        results_copy = results.copy()
        results_copy['trades'] = []
        for trade in self.trades:
            trade_copy = trade.copy()
            trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
            trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
            results_copy['trades'].append(trade_copy)

        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"💾 Результаты сохранены в {filename}")