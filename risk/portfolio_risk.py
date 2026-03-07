"""
Модуль управления портфельным риском
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from core.config import config


class PortfolioRiskManager:
    """Управление рисками на уровне портфеля"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Позиции по инструментам
        self.positions = {}  # {instrument: position_info}

        # История
        self.equity_curve = []  # [(timestamp, equity)]
        self.drawdowns = []
        self.trades = []

        # Лимиты
        self.max_portfolio_risk = config.get('trading', 'max_portfolio_risk_pct', default=0.10)
        self.max_correlation = 0.7  # Максимальная корреляция между позициями
        self.max_sector_exposure = 0.3  # Максимальная доля одного сектора

        # Ежедневные лимиты
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_loss_limit = config.get('risk', 'max_daily_loss_pct', default=0.05) * initial_capital

        # Статистика
        self.var_95 = 0.0  # Value at Risk (95%)
        self.expected_shortfall = 0.0
        self.beta_to_market = 1.0

    def add_position(self, instrument: str, position_info: Dict):
        """
        Добавляет новую позицию в портфель

        Args:
            instrument: тикер инструмента
            position_info: информация о позиции

        Returns:
            True если позиция добавлена, False если превышены лимиты
        """
        # Проверяем общий риск портфеля
        if not self._check_portfolio_risk(instrument, position_info):
            return False

        # Проверяем дневной лимит убытков
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            return False

        self.positions[instrument] = {
            **position_info,
            'entry_time': datetime.now(),
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0
        }

        return True

    def _check_portfolio_risk(self, instrument: str, new_position: Dict) -> bool:
        """
        Проверяет, не превышает ли новая позиция портфельные лимиты
        """
        # 1. Расчет общей экспозиции
        total_exposure = sum(p['position_value'] for p in self.positions.values())
        new_exposure = total_exposure + new_position['position_value']

        if new_exposure > self.current_capital * (1 + self.max_portfolio_risk):
            return False

        # 2. Проверка концентрации
        for existing_instrument, existing_pos in self.positions.items():
            # Если инструменты сильно коррелированы, проверяем общую долю
            correlation = self._get_correlation(instrument, existing_instrument)
            if correlation > self.max_correlation:
                combined_value = existing_pos['position_value'] + new_position['position_value']
                if combined_value > self.current_capital * 0.4:  # максимум 40% в коррелированные активы
                    return False

        # 3. Проверка дневного лимита убытков с учетом новой позиции
        potential_loss = new_position['risk_amount']
        if abs(self.daily_pnl) + potential_loss > self.daily_loss_limit:
            return False

        return True

    def _get_correlation(self, inst1: str, inst2: str) -> float:
        """
        Получает корреляцию между инструментами
        В реальном проекте нужно рассчитывать на основе исторических данных
        """
        # Заглушка - в реальном проекте здесь должен быть расчет корреляции
        return 0.5

    def update_positions(self, market_data: Dict[str, float]):
        """
        Обновляет стоимость открытых позиций

        Args:
            market_data: словарь {instrument: current_price}
        """
        total_value = self.current_capital

        for instrument, position in list(self.positions.items()):
            if instrument in market_data:
                current_price = market_data[instrument]
                entry_price = position['entry_price']
                size = position['position_size']

                # Расчет нереализованной PnL
                if position['direction'] == 'LONG':
                    unrealized = (current_price - entry_price) * size
                    unrealized_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized = (entry_price - current_price) * size
                    unrealized_pct = (entry_price - current_price) / entry_price

                position['unrealized_pnl'] = unrealized
                position['unrealized_pnl_pct'] = unrealized_pct
                position['current_price'] = current_price

                total_value += unrealized

        # Обновляем кривую капитала
        self.current_capital = total_value
        self.peak_capital = max(self.peak_capital, self.current_capital)

        # Рассчитываем просадку
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.drawdowns.append((datetime.now(), drawdown))

        self.equity_curve.append((datetime.now(), self.current_capital))

        # Обновляем VaR
        self._calculate_var()

    def _calculate_var(self, confidence: float = 0.95):
        """
        Рассчитывает Value at Risk на основе исторической волатильности
        """
        if len(self.equity_curve) < 20:
            return

        # Берем последние 100 точек кривой капитала
        recent_values = [v for _, v in self.equity_curve[-100:]]
        returns = pd.Series(recent_values).pct_change().dropna()

        if len(returns) > 0:
            self.var_95 = np.percentile(returns, (1 - confidence) * 100)
            self.expected_shortfall = returns[returns <= self.var_95].mean()

    def close_position(self, instrument: str, exit_price: float, reason: str) -> Dict:
        """
        Закрывает позицию и фиксирует PnL

        Args:
            instrument: тикер инструмента
            exit_price: цена выхода
            reason: причина закрытия

        Returns:
            Информация о закрытой позиции
        """
        if instrument not in self.positions:
            return None

        position = self.positions[instrument]

        # Расчет финальной PnL
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['position_size']
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - exit_price) * position['position_size']
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        # Вычитаем комиссию
        commission = config.get('trading', 'commission', default=0.001)
        pnl -= position['position_value'] * commission

        # Обновляем капитал
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.daily_trades += 1

        # Сохраняем сделку
        trade_record = {
            'instrument': instrument,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'position_value': position['position_value'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }

        self.trades.append(trade_record)

        # Удаляем позицию
        del self.positions[instrument]

        return trade_record

    def get_portfolio_status(self) -> Dict:
        """
        Возвращает текущий статус портфеля
        """
        total_exposure = sum(p['position_value'] for p in self.positions.values())
        unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.values())

        # Расчет текущей просадки
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Максимальная просадка за все время
        max_drawdown = max([dd for _, dd in self.drawdowns]) if self.drawdowns else 0

        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / self.current_capital if self.current_capital > 0 else 0,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall
        }

    def get_win_rate(self, period_days: int = 30) -> float:
        """
        Рассчитывает винрейт за указанный период
        """
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_trades = [t for t in self.trades if t['exit_time'] >= cutoff]

        if not recent_trades:
            return 0.0

        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        return wins / len(recent_trades)

    def reset_daily_stats(self):
        """Сбрасывает дневную статистику (вызывать в начале каждого дня)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def check_circuit_breakers(self) -> Dict:
        """
        Проверяет срабатывание защитных механизмов

        Returns:
            Словарь с результатами проверки
        """
        reasons = []
        should_stop = False

        # Проверка максимальной просадки
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > 0.15:  # 15% просадка
            should_stop = True
            reasons.append(f"Max drawdown exceeded: {current_drawdown:.2%}")

        # Проверка дневного лимита убытков
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            should_stop = True
            reasons.append(f"Daily loss limit reached: {self.daily_pnl:,.0f}")

        # Проверка VaR
        if self.var_95 < -0.05:  # 5% VaR
            reasons.append(f"High VaR: {self.var_95:.2%}")
            # Не останавливаем, но предупреждаем

        return {
            'should_stop': should_stop,
            'reasons': reasons
        }