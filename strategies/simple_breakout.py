"""
Простая пробойная стратегия
"""
import pandas as pd
import numpy as np
from typing import Dict

from strategies.base_strategy import BaseStrategy


class SimpleBreakoutStrategy(BaseStrategy):
    """
    Торговля на пробое диапазона
    """

    def __init__(self, instrument: str):
        super().__init__("Breakout", instrument)

        # Параметры
        self.range_period = 20  # период для определения диапазона
        self.breakout_threshold = 0.002  # 0.2% для подтверждения пробоя
        self.min_volume_ratio = 1.5  # объем должен быть в 1.5 раза выше среднего

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Для совместимости"""
        return df

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Сигнал: пробой максимума/минимума за N периодов с подтверждением объемом
        """
        if len(df) < self.range_period + 5:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}

        # Находим диапазон
        high_max = df['high'].iloc[-self.range_period:-1].max()
        low_min = df['low'].iloc[-self.range_period:-1].min()

        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]

        # Средний объем
        avg_volume = df['volume'].iloc[-self.range_period:-1].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # ATR для стопов
        atr = self._calculate_atr(df)

        result = {
            'signal': 'HOLD',
            'confidence': 0,
            'entry_price': current_close,
            'stop_loss': None,
            'take_profit': None,
            'reason': []
        }

        # Пробой вверх
        if current_high > high_max * (1 + self.breakout_threshold):
            if volume_ratio > self.min_volume_ratio:
                result['signal'] = 'BUY'
                result['confidence'] = min(0.9, volume_ratio / 3)
                result['stop_loss'] = high_max * 0.995  # чуть ниже уровня пробоя
                result['take_profit'] = current_close + atr * 3
                result['reason'].append(f"Upside breakout above {high_max:.2f}")
                result['reason'].append(f"Volume: {volume_ratio:.1f}x average")

        # Пробой вниз
        elif current_low < low_min * (1 - self.breakout_threshold):
            if volume_ratio > self.min_volume_ratio:
                result['signal'] = 'SELL'
                result['confidence'] = min(0.9, volume_ratio / 3)
                result['stop_loss'] = low_min * 1.005  # чуть выше уровня пробоя
                result['take_profit'] = current_close - atr * 3
                result['reason'].append(f"Downside breakout below {low_min:.2f}")
                result['reason'].append(f"Volume: {volume_ratio:.1f}x average")

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Рассчитывает ATR"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0

    def get_params(self) -> Dict:
        return {
            'range_period': self.range_period,
            'breakout_threshold': self.breakout_threshold,
            'min_volume_ratio': self.min_volume_ratio
        }

    def set_params(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)