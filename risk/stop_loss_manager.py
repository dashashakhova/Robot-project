"""
Модуль управления стоп-лоссами и тейк-профитами
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from core.config import config


class StopLossType(Enum):
    """Типы стоп-лоссов"""
    FIXED = "fixed"  # Фиксированный процент
    ATR = "atr"  # На основе ATR
    TRAILING = "trailing"  # Трейлинг стоп
    TIME = "time"  # По времени
    SUPPORT = "support"  # За уровнем поддержки/сопротивления


class StopLossManager:
    """Управление стоп-лоссами и тейк-профитами"""

    def __init__(self, instrument: str):
        self.instrument = instrument
        self.active_stops = {}  # Словарь активных стопов по сделкам

        # Параметры по умолчанию
        self.default_stop_type = StopLossType.ATR
        self.default_atr_multiplier = 1.5
        self.default_trailing_activation = 0.5  # Активация трейлинга при прибыли 0.5%
        self.default_trailing_distance = 0.3  # Расстояние трейлинга 0.3%
        self.max_hold_hours = 8  # Максимальное время удержания

    def calculate_stop_loss(self, entry_price: float, direction: str,
                            atr: Optional[float] = None,
                            support_levels: Optional[List[float]] = None,
                            resistance_levels: Optional[List[float]] = None,
                            stop_type: Optional[StopLossType] = None) -> Dict:
        """
        Рассчитывает уровень стоп-лосса

        Args:
            entry_price: цена входа
            direction: направление ('LONG' или 'SHORT')
            atr: значение ATR
            support_levels: уровни поддержки
            resistance_levels: уровни сопротивления
            stop_type: тип стоп-лосса

        Returns:
            Словарь с параметрами стопа
        """
        if stop_type is None:
            stop_type = self.default_stop_type

        stop_price = None
        stop_percent = None
        reason = ""

        if stop_type == StopLossType.FIXED:
            # Фиксированный процент
            risk_percent = config.get('risk', 'max_risk_per_trade', default=0.02)
            if direction == 'LONG':
                stop_price = entry_price * (1 - risk_percent)
                stop_percent = -risk_percent * 100
            else:
                stop_price = entry_price * (1 + risk_percent)
                stop_percent = -risk_percent * 100
            reason = f"Fixed {risk_percent:.1%} stop"

        elif stop_type == StopLossType.ATR and atr:
            # На основе ATR
            atr_value = atr * self.default_atr_multiplier
            if direction == 'LONG':
                stop_price = entry_price - atr_value
                stop_percent = -(atr_value / entry_price) * 100
            else:
                stop_price = entry_price + atr_value
                stop_percent = -(atr_value / entry_price) * 100
            reason = f"ATR-based stop ({self.default_atr_multiplier}x ATR)"

        elif stop_type == StopLossType.SUPPORT and direction == 'LONG' and support_levels:
            # За уровнем поддержки
            valid_supports = [s for s in support_levels if s < entry_price]
            if valid_supports:
                # Берем ближайший уровень поддержки
                nearest_support = max(valid_supports)
                stop_price = nearest_support * 0.995  # Немного ниже уровня
                stop_percent = ((stop_price - entry_price) / entry_price) * 100
                reason = f"Below support at {nearest_support:.2f}"
            else:
                # Если нет поддержки, используем ATR
                return self.calculate_stop_loss(entry_price, direction, atr,
                                                stop_type=StopLossType.ATR)

        elif stop_type == StopLossType.SUPPORT and direction == 'SHORT' and resistance_levels:
            # За уровнем сопротивления
            valid_resistances = [r for r in resistance_levels if r > entry_price]
            if valid_resistances:
                # Берем ближайший уровень сопротивления
                nearest_resistance = min(valid_resistances)
                stop_price = nearest_resistance * 1.005  # Немного выше уровня
                stop_percent = ((entry_price - stop_price) / entry_price) * 100
                reason = f"Above resistance at {nearest_resistance:.2f}"
            else:
                return self.calculate_stop_loss(entry_price, direction, atr,
                                                stop_type=StopLossType.ATR)

        if stop_price is None:
            # Fallback на фиксированный стоп
            return self.calculate_stop_loss(entry_price, direction,
                                            stop_type=StopLossType.FIXED)

        return {
            'stop_price': stop_price,
            'stop_percent': stop_percent,
            'stop_type': stop_type.value,
            'reason': reason
        }

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                              direction: str, min_rr: float = None) -> Dict:
        """
        Рассчитывает уровень тейк-профита

        Args:
            entry_price: цена входа
            stop_loss: цена стоп-лосса
            direction: направление
            min_rr: минимальное соотношение риск/прибыль

        Returns:
            Словарь с параметрами тейка
        """
        if min_rr is None:
            min_rr = config.get('risk', 'min_risk_reward', default=1.5)

        risk = abs(entry_price - stop_loss)

        if direction == 'LONG':
            take_profit = entry_price + (risk * min_rr)
        else:
            take_profit = entry_price - (risk * min_rr)

        profit_percent = ((take_profit - entry_price) / entry_price) * 100
        if direction == 'SHORT':
            profit_percent = -profit_percent

        return {
            'take_profit': take_profit,
            'profit_percent': profit_percent,
            'risk_reward_ratio': min_rr
        }

    def create_stop_order(self, trade_id: str, entry_price: float, direction: str,
                          atr: Optional[float] = None,
                          support_levels: Optional[List[float]] = None,
                          resistance_levels: Optional[List[float]] = None) -> Dict:
        """
        Создает стоп-ордер для сделки

        Args:
            trade_id: идентификатор сделки
            entry_price: цена входа
            direction: направление
            atr: ATR для динамического стопа
            support_levels: уровни поддержки
            resistance_levels: уровни сопротивления

        Returns:
            Словарь с параметрами стоп-ордера
        """
        # Рассчитываем стоп
        stop_info = self.calculate_stop_loss(
            entry_price, direction, atr, support_levels, resistance_levels
        )

        # Рассчитываем тейк
        take_profit_info = self.calculate_take_profit(
            entry_price, stop_info['stop_price'], direction
        )

        # Сохраняем информацию о стопе
        stop_order = {
            'trade_id': trade_id,
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_info['stop_price'],
            'take_profit': take_profit_info['take_profit'],
            'stop_type': stop_info['stop_type'],
            'stop_reason': stop_info['reason'],
            'created_at': datetime.now(),
            'is_trailing': False,
            'trailing_activated': False,
            'trailing_distance': self.default_trailing_distance,
            'trailing_activation': self.default_trailing_activation
        }

        self.active_stops[trade_id] = stop_order
        return stop_order

    def update_stops(self, trade_id: str, current_price: float,
                     highest_price: float = None, lowest_price: float = None) -> Dict:
        """
        Обновляет стопы для сделки (трейлинг стоп)

        Args:
            trade_id: идентификатор сделки
            current_price: текущая цена
            highest_price: максимальная цена с момента входа (для LONG)
            lowest_price: минимальная цена с момента входа (для SHORT)

        Returns:
            Обновленный стоп-ордер или None если стоп не изменился
        """
        if trade_id not in self.active_stops:
            return None

        stop = self.active_stops[trade_id]

        # Проверяем время удержания
        hold_time = datetime.now() - stop['created_at']
        if hold_time.total_seconds() > self.max_hold_hours * 3600:
            return {
                'action': 'CLOSE',
                'reason': 'Max hold time reached',
                'price': current_price
            }

        # Для LONG позиций
        if stop['direction'] == 'LONG' and highest_price:
            # Расчет прибыли от максимума
            profit_from_high = (highest_price - stop['entry_price']) / stop['entry_price'] * 100

            # Активация трейлинг стопа
            if not stop['trailing_activated'] and profit_from_high >= stop['trailing_activation']:
                stop['trailing_activated'] = True
                stop['is_trailing'] = True
                stop['trailing_high'] = highest_price

            # Обновление трейлинг стопа
            if stop['trailing_activated'] and highest_price > stop.get('trailing_high', 0):
                stop['trailing_high'] = highest_price
                # Подтягиваем стоп
                new_stop = highest_price * (1 - stop['trailing_distance'] / 100)
                if new_stop > stop['stop_loss']:
                    stop['stop_loss'] = new_stop
                    return {
                        'action': 'UPDATE',
                        'new_stop': new_stop,
                        'reason': 'Trailing stop updated'
                    }

        # Для SHORT позиций
        elif stop['direction'] == 'SHORT' and lowest_price:
            profit_from_low = (stop['entry_price'] - lowest_price) / stop['entry_price'] * 100

            if not stop['trailing_activated'] and profit_from_low >= stop['trailing_activation']:
                stop['trailing_activated'] = True
                stop['is_trailing'] = True
                stop['trailing_low'] = lowest_price

            if stop['trailing_activated'] and lowest_price < stop.get('trailing_low', float('inf')):
                stop['trailing_low'] = lowest_price
                new_stop = lowest_price * (1 + stop['trailing_distance'] / 100)
                if new_stop < stop['stop_loss']:
                    stop['stop_loss'] = new_stop
                    return {
                        'action': 'UPDATE',
                        'new_stop': new_stop,
                        'reason': 'Trailing stop updated'
                    }

        return None

    def check_stop_trigger(self, trade_id: str, current_price: float) -> Dict:
        """
        Проверяет, сработал ли стоп-лосс или тейк-профит

        Args:
            trade_id: идентификатор сделки
            current_price: текущая цена

        Returns:
            Словарь с результатом проверки
        """
        if trade_id not in self.active_stops:
            return {'triggered': False}

        stop = self.active_stops[trade_id]

        # Проверка стоп-лосса
        if stop['direction'] == 'LONG' and current_price <= stop['stop_loss']:
            return {
                'triggered': True,
                'type': 'STOP_LOSS',
                'price': stop['stop_loss'],
                'reason': 'Stop loss triggered'
            }
        elif stop['direction'] == 'SHORT' and current_price >= stop['stop_loss']:
            return {
                'triggered': True,
                'type': 'STOP_LOSS',
                'price': stop['stop_loss'],
                'reason': 'Stop loss triggered'
            }

        # Проверка тейк-профита
        if stop['direction'] == 'LONG' and current_price >= stop['take_profit']:
            return {
                'triggered': True,
                'type': 'TAKE_PROFIT',
                'price': stop['take_profit'],
                'reason': 'Take profit triggered'
            }
        elif stop['direction'] == 'SHORT' and current_price <= stop['take_profit']:
            return {
                'triggered': True,
                'type': 'TAKE_PROFIT',
                'price': stop['take_profit'],
                'reason': 'Take profit triggered'
            }

        return {'triggered': False}

    def remove_stop(self, trade_id: str):
        """Удаляет стоп для завершенной сделки"""
        if trade_id in self.active_stops:
            del self.active_stops[trade_id]