"""
Модуль для расчета размера позиции на основе риска
"""
from typing import Dict, Optional
from datetime import datetime

from core.config import config


class PositionSizeManager:
    """Управление размером позиции на основе риска"""

    def __init__(self, capital: float):
        """
        Args:
            capital: текущий капитал
        """
        self.capital = capital
        self.max_risk_per_trade = config.get('risk', 'max_risk_per_trade', default=0.02)
        self.max_position_pct = config.get('trading', 'max_position_size_pct', default=0.25)
        self.max_daily_loss_pct = config.get('risk', 'max_daily_loss_pct', default=0.05)

        # История для отслеживания дневных убытков
        self.current_day = datetime.now().date()
        self.daily_loss = 0.0  # Накопленный убыток за день (только downside, всегда >= 0)

        # Статистика
        self.trade_history = []
        self.consecutive_losses = 0

    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                                direction: str, atr: Optional[float] = None) -> Dict:
        """
        Рассчитывает оптимальный размер позиции

        Args:
            entry_price: цена входа
            stop_loss_price: цена стоп-лосса
            direction: направление ('LONG' или 'SHORT')
            atr: средний истинный диапазон (для альтернативного расчета)

        Returns:
            Словарь с параметрами позиции
        """
        # Проверяем дневной лимит убытков
        if not self._check_daily_loss_limit():
            return {
                'can_trade': False,
                'reason': 'Daily loss limit reached',
                'position_size': 0,
                'risk_amount': 0
            }

        # Проверяем лимит на последовательные убытки
        if self.consecutive_losses >= config.get('risk', 'max_consecutive_losses', default=3):
            return {
                'can_trade': False,
                'reason': 'Max consecutive losses reached',
                'position_size': 0,
                'risk_amount': 0
            }

        # Рассчитываем риск в рублях
        risk_per_share = abs(entry_price - stop_loss_price)

        # Максимальный риск в рублях (2% от капитала)
        max_risk_rub = self.capital * self.max_risk_per_trade

        # Количество акций исходя из риска
        if risk_per_share > 0:
            shares_by_risk = max_risk_rub / risk_per_share
        else:
            shares_by_risk = 0

        # Максимальное количество акций исходя из лимита доли капитала
        position_pct = min(max(self.max_position_pct, 0.0), 1.0)
        if position_pct == 0:
            return {
                'can_trade': False,
                'reason': 'Invalid max_position_size_pct',
                'position_size': 0,
                'risk_amount': 0
            }

        max_position_value = self.capital * position_pct
        shares_by_capital = max_position_value / entry_price

        # Итоговое количество акций - минимум из двух ограничений
        shares = min(shares_by_risk, shares_by_capital)

        # Округляем до целого количества лотов
        lot_size = self._get_lot_size()  # нужно получить из API или конфига
        if lot_size > 0:
            shares = int(shares / lot_size) * lot_size

        position_value = shares * entry_price
        risk_amount = shares * risk_per_share
        risk_pct = risk_amount / self.capital if self.capital > 0 else 0

        # Проверяем минимальный размер позиции
        min_position_value = self.capital * 0.01  # минимум 1% от капитала
        if position_value < min_position_value:
            return {
                'can_trade': False,
                'reason': 'Position too small',
                'position_size': 0,
                'risk_amount': 0
            }

        # Рассчитываем потенциальную прибыль (предполагаем risk/reward = 2)
        target_price = self._calculate_target_price(entry_price, risk_per_share, direction)

        return {
            'can_trade': True,
            'reason': 'OK',
            'position_size': int(shares),
            'position_value': position_value,
            'position_value_pct': position_value / self.capital,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'target_price': target_price,
            'risk_reward_ratio': abs(target_price - entry_price) / risk_per_share if risk_per_share > 0 else 0
        }

    def _check_daily_loss_limit(self) -> bool:
        """Проверяет, не превышен ли дневной лимит убытков"""
        today = datetime.now().date()

        # Сбрасываем счетчик для нового дня
        if today != self.current_day:
            self.current_day = today
            self.daily_loss = 0.0

        # Проверяем лимит
        max_daily_loss = self.capital * self.max_daily_loss_pct
        return self.daily_loss < max_daily_loss

    def _calculate_target_price(self, entry_price: float, risk_per_share: float,
                                direction: str) -> float:
        """Рассчитывает цену тейк-профита (риск/прибыль = 2)"""
        min_rr = config.get('risk', 'min_risk_reward', default=1.5)

        if direction == 'LONG':
            return entry_price + (risk_per_share * min_rr)
        else:
            return entry_price - (risk_per_share * min_rr)

    def _get_lot_size(self) -> int:
        """Получает размер лота для инструмента"""
        # В реальном проекте нужно получать из API
        # Пока возвращаем 1 (для большинства акций)
        return 1

    def update_trade_result(self, trade_result: Dict):
        """Обновляет статистику после сделки"""
        self.trade_history.append(trade_result)

        # Обновляем дневной убыток: учитываем только отрицательные сделки
        pnl = trade_result.get('pnl', 0)
        if pnl < 0:
            self.daily_loss += abs(pnl)

        # Обновляем счетчик последовательных убытков
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def adjust_for_volatility(self, position_size: Dict, atr_percent: float) -> Dict:
        """
        Корректирует размер позиции с учетом волатильности

        Args:
            position_size: исходный расчет позиции
            atr_percent: текущая волатильность в процентах

        Returns:
            Скорректированный размер позиции
        """
        if not position_size['can_trade']:
            return position_size

        # Базовая волатильность (норма)
        base_volatility = 1.0  # 1% дневного движения

        # Коэффициент коррекции
        if atr_percent > base_volatility * 2:
            # Очень высокая волатильность - уменьшаем позицию на 50%
            adjustment = 0.5
        elif atr_percent > base_volatility * 1.5:
            # Высокая волатильность - уменьшаем на 25%
            adjustment = 0.75
        elif atr_percent < base_volatility * 0.5:
            # Низкая волатильность - можно увеличить на 20%
            adjustment = 1.2
        else:
            adjustment = 1.0

        # Применяем коррекцию
        position_size['position_size'] = int(position_size['position_size'] * adjustment)
        position_size['position_value'] *= adjustment
        position_size['risk_amount'] *= adjustment
        position_size['risk_pct'] *= adjustment
        position_size['adjustment_factor'] = adjustment
        position_size['adjustment_reason'] = f'Volatility adjustment: {adjustment:.2f}x'

        return position_size

    def get_status(self) -> Dict:
        """Возвращает текущий статус риск-менеджера"""
        return {
            'capital': self.capital,
            'max_risk_per_trade': self.max_risk_per_trade,
            'daily_loss': self.daily_loss,
            'daily_loss_limit': self.capital * self.max_daily_loss_pct,
            'consecutive_losses': self.consecutive_losses,
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate()
        }

    def _calculate_win_rate(self) -> float:
        """Рассчитывает винрейт по последним сделкам"""
        if len(self.trade_history) < 10:
            return 0.0

        last_trades = self.trade_history[-20:]  # последние 20 сделок
        wins = sum(1 for t in last_trades if t.get('pnl', 0) > 0)
        return wins / len(last_trades) if last_trades else 0.0
