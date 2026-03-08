"""
Стратегия торговли от уровней поддержки/сопротивления
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from strategies.base_strategy import BaseStrategy


class LevelTradingStrategy(BaseStrategy):
    """
    Торговля от уровней поддержки/сопротивления с контролем риска
    """

    def __init__(self, instrument: str):
        super().__init__("LevelTrader", instrument)

        # Параметры стратегии
        self.lookback_periods = 100  # свечей для поиска уровней
        self.level_threshold = 0.001  # 0.1% кластеризация уровней
        self.bounce_threshold = 0.002  # 0.2% отскок от уровня
        self.breakout_threshold = 0.003  # 0.3% пробой уровня

        # Риск-менеджмент
        self.stop_loss_pct = 0.005  # 0.5% стоп
        self.take_profit_pct = 0.01  # 1% тейк
        self.max_risk_per_trade = 0.02  # 2% риска на сделку

        # Состояние
        self.levels = {'support': [], 'resistance': []}
        self.last_levels_update = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Для совместимости с базовым классом"""
        return df

    def find_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Находит уровни поддержки и сопротивления
        """
        high = df['high'].values[-self.lookback_periods:]
        low = df['low'].values[-self.lookback_periods:]
        close = df['close'].values[-1]

        # Ищем локальные максимумы и минимумы
        resistance_candidates = []
        support_candidates = []

        for i in range(5, len(high) - 5):
            # Локальный максимум
            if high[i] == max(high[i - 5:i + 5]):
                resistance_candidates.append(high[i])

            # Локальный минимум
            if low[i] == min(low[i - 5:i + 5]):
                support_candidates.append(low[i])

        # Кластеризация близких уровней
        resistance = self._cluster_levels(resistance_candidates)
        support = self._cluster_levels(support_candidates)

        # Оставляем только значимые уровни (не слишком далеко от цены)
        max_distance = close * 0.05  # 5% от текущей цены
        resistance = [r for r in resistance if abs(r - close) < max_distance]
        support = [s for s in support if abs(s - close) < max_distance]

        return {
            'resistance': sorted(resistance),
            'support': sorted(support, reverse=True)
        }

    def _cluster_levels(self, candidates: List[float]) -> List[float]:
        """Группирует близкие уровни"""
        if not candidates:
            return []

        candidates = sorted(candidates)
        clustered = []
        current_cluster = [candidates[0]]

        for c in candidates[1:]:
            # Если расстояние меньше порога, объединяем
            if abs(c - current_cluster[-1]) / current_cluster[-1] < self.level_threshold:
                current_cluster.append(c)
            else:
                # Берем среднее по кластеру
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [c]

        clustered.append(sum(current_cluster) / len(current_cluster))
        return clustered

    def analyze_price_action(self, df: pd.DataFrame, levels: Dict[str, List[float]]) -> Dict:
        """
        Анализирует ценовое действие относительно уровней
        """
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = (current_price - prev_price) / prev_price

        result = {
            'signal': 'HOLD',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'reason': []
        }

        # Проверяем подход к уровням сопротивления
        for level in levels['resistance']:
            distance_to_resistance = abs(level - current_price) / current_price

            # Если цена подошла к сопротивлению
            if distance_to_resistance < self.bounce_threshold:
                # Проверяем на отскок вниз
                if price_change < 0:  # Цена уже идет вниз
                    result['signal'] = 'SELL'
                    result['confidence'] = min(1.0, self.bounce_threshold / (distance_to_resistance + 0.0001))
                    result['stop_loss'] = level * 1.002  # Стоп чуть выше уровня
                    result['take_profit'] = current_price * (1 - self.take_profit_pct)
                    result['reason'].append(f"Resistance bounce at {level:.2f}")
                    break

                # Проверяем на пробой вверх
                elif price_change > self.breakout_threshold:
                    result['signal'] = 'BUY'
                    result['confidence'] = 0.8
                    result['stop_loss'] = level * 0.998  # Стоп чуть ниже уровня
                    result['take_profit'] = current_price * (1 + self.take_profit_pct)
                    result['reason'].append(f"Resistance breakout at {level:.2f}")
                    break

        # Проверяем подход к уровням поддержки
        if result['signal'] == 'HOLD':
            for level in levels['support']:
                distance_to_support = abs(level - current_price) / current_price

                if distance_to_support < self.bounce_threshold:
                    if price_change > 0:  # Цена уже идет вверх
                        result['signal'] = 'BUY'
                        result['confidence'] = min(1.0, self.bounce_threshold / (distance_to_support + 0.0001))
                        result['stop_loss'] = level * 0.998  # Стоп чуть ниже уровня
                        result['take_profit'] = current_price * (1 + self.take_profit_pct)
                        result['reason'].append(f"Support bounce at {level:.2f}")
                        break

                    elif price_change < -self.breakout_threshold:
                        result['signal'] = 'SELL'
                        result['confidence'] = 0.8
                        result['stop_loss'] = level * 1.002  # Стоп чуть выше уровня
                        result['take_profit'] = current_price * (1 - self.take_profit_pct)
                        result['reason'].append(f"Support breakdown at {level:.2f}")
                        break

        return result

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Получает торговый сигнал
        """
        if len(df) < self.lookback_periods:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': ['Insufficient data']}

        # Обновляем уровни
        self.levels = self.find_levels(df)

        # Анализируем ценовое действие
        analysis = self.analyze_price_action(df, self.levels)

        # Добавляем метаданные
        if analysis['signal'] != 'HOLD':
            analysis['levels'] = self.levels
            analysis['current_price'] = df['close'].iloc[-1]
            analysis['timestamp'] = df.index[-1]
            analysis['strategy'] = self.name

            # Рассчитываем размер позиции (упрощенно)
            risk_amount = self.max_risk_per_trade * 100000  # от капитала 100k
            stop_distance = abs(analysis['entry_price'] - analysis['stop_loss'])
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0

            analysis['position_size'] = int(position_size)
            analysis['risk_amount'] = risk_amount

        return analysis

    def get_params(self) -> Dict:
        return {
            'lookback_periods': self.lookback_periods,
            'level_threshold': self.level_threshold,
            'bounce_threshold': self.bounce_threshold,
            'breakout_threshold': self.breakout_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_risk_per_trade': self.max_risk_per_trade
        }

    def set_params(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)