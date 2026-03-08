"""
Простая трендовая стратегия на основе скользящих средних
"""
import pandas as pd
import numpy as np
from typing import Dict, List

from strategies.base_strategy import BaseStrategy


class TrendFollowerStrategy(BaseStrategy):
    """
    Следование за трендом с использованием двух скользящих средних
    """

    def __init__(self, instrument: str):
        super().__init__("TrendFollower", instrument)

        # Параметры
        self.fast_ma = 20
        self.slow_ma = 50
        self.volume_ma = 20
        self.atr_period = 14
        self.atr_multiplier = 2.0

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Для совместимости"""
        return df

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Сигнал: пересечение скользящих средних
        """
        if len(df) < self.slow_ma + 10:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}

        # Рассчитываем индикаторы
        fast_ma = df['close'].rolling(self.fast_ma).mean()
        slow_ma = df['close'].rolling(self.slow_ma).mean()
        volume_ma = df['volume'].rolling(self.volume_ma).mean()
        atr = self._calculate_atr(df)

        current_price = df['close'].iloc[-1]
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]

        # Проверка объема (подтверждение тренда)
        volume_ok = df['volume'].iloc[-1] > volume_ma.iloc[-1] * 1.2

        result = {
            'signal': 'HOLD',
            'confidence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'reason': []
        }

        # Сигнал на покупку: быстрая пересекает медленную снизу вверх
        if prev_fast <= prev_slow and current_fast > current_slow:
            if volume_ok:
                result['signal'] = 'BUY'
                result['confidence'] = 0.7 if volume_ok else 0.5
                result['stop_loss'] = current_price - atr * self.atr_multiplier
                result['take_profit'] = current_price + atr * self.atr_multiplier * 2
                result['reason'].append(f"Golden cross: {self.fast_ma} EMA crossed {self.slow_ma} EMA")
                if volume_ok:
                    result['reason'].append("Volume confirmation")

        # Сигнал на продажу: быстрая пересекает медленную сверху вниз
        elif prev_fast >= prev_slow and current_fast < current_slow:
            if volume_ok:
                result['signal'] = 'SELL'
                result['confidence'] = 0.7 if volume_ok else 0.5
                result['stop_loss'] = current_price + atr * self.atr_multiplier
                result['take_profit'] = current_price - atr * self.atr_multiplier * 2
                result['reason'].append(f"Death cross: {self.fast_ma} EMA crossed below {self.slow_ma} EMA")
                if volume_ok:
                    result['reason'].append("Volume confirmation")

        return result

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Рассчитывает ATR"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0

    def get_params(self) -> Dict:
        return {
            'fast_ma': self.fast_ma,
            'slow_ma': self.slow_ma,
            'volume_ma': self.volume_ma,
            'atr_multiplier': self.atr_multiplier
        }

    def set_params(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)