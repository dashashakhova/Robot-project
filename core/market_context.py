"""
Модуль анализа рыночного контекста и режимов рынка
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta


class MarketRegime(Enum):
    """Режимы рынка"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MarketContextAnalyzer:
    """Анализирует общее состояние рынка для конкретного инструмента"""

    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.regime_history = []
        self.current_regime = MarketRegime.RANGING
        self.regime_confidence = 0.0
        self.metrics = {}

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Проводит полный анализ рыночного контекста

        Args:
            df: DataFrame с индикаторами

        Returns:
            Словарь с результатами анализа
        """
        if len(df) < self.lookback_periods:
            df = df.copy()

        # Получаем последние данные
        recent = df.iloc[-self.lookback_periods:].copy()

        # 1. Определяем тренд
        trend_analysis = self._analyze_trend(recent)

        # 2. Анализируем волатильность
        volatility_analysis = self._analyze_volatility(recent)

        # 3. Анализируем объемы
        volume_analysis = self._analyze_volume(recent)

        # 4. Определяем фазу рынка
        regime, confidence = self._determine_regime(
            trend_analysis, volatility_analysis, volume_analysis
        )

        # 5. Анализируем поддержку/сопротивление
        support_resistance = self._find_support_resistance(recent)

        # Сохраняем результаты
        self.current_regime = regime
        self.regime_confidence = confidence
        self.regime_history.append({
            'time': df.index[-1],
            'regime': regime.value,
            'confidence': confidence
        })

        # Ограничиваем историю
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        self.metrics = {
            'trend': trend_analysis,
            'volatility': volatility_analysis,
            'volume': volume_analysis,
            'regime': regime.value,
            'regime_confidence': confidence,
            'support_resistance': support_resistance,
            'should_trade': self.should_trade()
        }

        return self.metrics

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Анализирует тренд с помощью нескольких методов"""
        close = df['close']

        # 1. Скользящие средние
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else close.rolling(20).mean().iloc[-1]
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else close.rolling(50).mean().iloc[-1]
        sma_100 = df['sma_100'].iloc[-1] if 'sma_100' in df.columns else close.rolling(100).mean().iloc[-1]

        # 2. ADX (Average Directional Index)
        adx = self._calculate_adx(df)

        # 3. MACD
        macd_line = df['macd_line'].iloc[-1] if 'macd_line' in df.columns else 0
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0

        # 4. Линейная регрессия для определения наклона
        x = np.arange(len(close[-20:]))
        y = close[-20:].values
        slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        slope_pct = (slope / close.iloc[-1]) * 100 if close.iloc[-1] != 0 else 0

        # 5. Процент баров выше/ниже EMA
        ema_20 = df['ema_20'].iloc[-1] if 'ema_20' in df.columns else close.ewm(span=20).mean().iloc[-1]
        above_ema = (close[-20:] > ema_20).sum() / 20

        return {
            'price_vs_sma20': (close.iloc[-1] - sma_20) / sma_20,
            'price_vs_sma50': (close.iloc[-1] - sma_50) / sma_50,
            'price_vs_sma100': (close.iloc[-1] - sma_100) / sma_100,
            'sma20_vs_sma50': (sma_20 - sma_50) / sma_50,
            'sma50_vs_sma100': (sma_50 - sma_100) / sma_100,
            'adx': adx,
            'macd_histogram': macd_line - macd_signal,
            'slope_20': slope_pct,
            'above_ema_ratio': above_ema,
            'trend_strength': self._calculate_trend_strength(df)
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Рассчитывает ADX"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            # Сглаживание
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean().iloc[-1]

            return adx if not pd.isna(adx) else 0
        except:
            return 0

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Рассчитывает силу тренда (0-1)"""
        try:
            close = df['close']
            returns = close.pct_change().dropna()

            # Используем соотношение сигнал/шум
            signal = returns.mean() * len(returns)
            noise = returns.std() * np.sqrt(len(returns))

            if noise == 0:
                return 0

            snr = abs(signal / noise)
            # Нормализуем до 0-1
            strength = min(1.0, snr / 10)

            return strength
        except:
            return 0

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Анализирует волатильность"""
        if 'atr_percent' in df.columns:
            current_atr = df['atr_percent'].iloc[-1]
            atr_history = df['atr_percent'].iloc[-20:]
        else:
            # Рассчитываем ATR если нет в данных
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = atr / df['close'] * 100
            current_atr = atr_pct.iloc[-1]
            atr_history = atr_pct.iloc[-20:]

        # Статистика волатильности
        volatility_stats = {
            'current_atr_pct': current_atr,
            'atr_percentile': (atr_history <= current_atr).sum() / len(atr_history) * 100,
            'atr_vs_ma': current_atr / atr_history.mean() if atr_history.mean() > 0 else 1,
            'is_high_volatility': current_atr > atr_history.quantile(0.8),
            'is_low_volatility': current_atr < atr_history.quantile(0.2)
        }

        # Волатильность доходностей
        returns = df['close'].pct_change().dropna()
        recent_returns = returns.iloc[-20:]

        volatility_stats.update({
            'returns_std': recent_returns.std() * 100,
            'returns_std_percentile': (returns.iloc[-100:].std() <= recent_returns.std()).sum() / 100 * 100
        })

        return volatility_stats

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Анализирует объемы"""
        volume = df['volume']

        if 'volume_sma' in df.columns:
            volume_sma = df['volume_sma']
        else:
            volume_sma = volume.rolling(20).mean()

        volume_ratio = volume / volume_sma
        current_ratio = volume_ratio.iloc[-1]

        # Тренд объема
        volume_trend = volume_ratio.iloc[-5:].mean() / volume_ratio.iloc[-20:-5].mean() if len(volume_ratio) > 20 else 1

        # Анализ объема в контексте движения цены
        price_change = df['close'].pct_change()
        volume_price_correlation = price_change.iloc[-20:].corr(volume_ratio.iloc[-20:])

        return {
            'volume_ratio': current_ratio,
            'volume_trend': volume_trend,
            'is_high_volume': current_ratio > 1.5,
            'is_low_volume': current_ratio < 0.5,
            'volume_price_correlation': volume_price_correlation,
            'volume_percentile': (volume_sma.iloc[-100:] <= volume.iloc[-1]).sum() / 100 * 100 if len(volume_sma) >= 100 else 50
        }

    def _determine_regime(self, trend: Dict, volatility: Dict, volume: Dict) -> Tuple[MarketRegime, float]:
        """
        Определяет режим рынка и уверенность в определении
        """
        confidence = 0.5  # Базовая уверенность
        regime = MarketRegime.RANGING

        # Оценка тренда
        trend_score = 0
        if trend['adx'] > 25:
            trend_score += 1
            confidence += 0.1

        if trend['price_vs_sma50'] > 0.02:
            trend_score += 1
        elif trend['price_vs_sma50'] < -0.02:
            trend_score -= 1

        if trend['sma20_vs_sma50'] > 0.01:
            trend_score += 1
        elif trend['sma20_vs_sma50'] < -0.01:
            trend_score -= 1

        if trend['macd_histogram'] > 0:
            trend_score += 0.5
        else:
            trend_score -= 0.5

        # Определяем режим на основе скора
        if abs(trend_score) >= 3:
            confidence += 0.2
            if trend_score > 0:
                regime = MarketRegime.STRONG_UPTREND
            else:
                regime = MarketRegime.STRONG_DOWNTREND
        elif abs(trend_score) >= 1.5:
            confidence += 0.1
            if trend_score > 0:
                regime = MarketRegime.WEAK_UPTREND
            else:
                regime = MarketRegime.WEAK_DOWNTREND
        else:
            regime = MarketRegime.RANGING

        # Учитываем волатильность
        if volatility['is_high_volatility']:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence += 0.15
        elif volatility['is_low_volatility'] and regime == MarketRegime.RANGING:
            regime = MarketRegime.LOW_VOLATILITY
            confidence += 0.1

        # Корректируем уверенность
        confidence = min(1.0, max(0.3, confidence))

        return regime, confidence

    def _find_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Находит уровни поддержки и сопротивления"""
        high = df['high'].iloc[-lookback:]
        low = df['low'].iloc[-lookback:]
        close = df['close'].iloc[-1]

        # Находим локальные максимумы и минимумы
        resistance_levels = []
        support_levels = []

        for i in range(5, len(high) - 5):
            if high.iloc[i] == high.iloc[i-5:i+5].max():
                resistance_levels.append(high.iloc[i])
            if low.iloc[i] == low.iloc[i-5:i+5].min():
                support_levels.append(low.iloc[i])

        # Группируем близкие уровни
        resistance = self._cluster_levels(resistance_levels)
        support = self._cluster_levels(support_levels)

        # Находим ближайшие уровни
        nearest_resistance = min([r for r in resistance if r > close], default=None)
        nearest_support = max([s for s in support if s < close], default=None)

        # Расстояние до уровней в процентах
        resistance_distance = (nearest_resistance - close) / close * 100 if nearest_resistance else None
        support_distance = (close - nearest_support) / close * 100 if nearest_support else None

        return {
            'resistance_levels': resistance[:3],  # Топ-3 уровня
            'support_levels': support[:3],
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_distance_pct': resistance_distance,
            'support_distance_pct': support_distance
        }

    def _cluster_levels(self, levels: List[float], threshold_pct: float = 0.5) -> List[float]:
        """Группирует близкие уровни"""
        if not levels:
            return []

        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            prev_avg = sum(current_cluster) / len(current_cluster)
            if abs(level - prev_avg) / prev_avg * 100 < threshold_pct:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        clustered.append(sum(current_cluster) / len(current_cluster))
        return clustered

    def should_trade(self) -> Dict:
        """
        Определяет, стоит ли торговать в текущем режиме

        Returns:
            Словарь с решением и причинами
        """
        reasons = []
        should_trade = True

        # Проверяем режим
        if self.current_regime in [MarketRegime.HIGH_VOLATILITY]:
            reasons.append("High volatility - reduce position size")
            should_trade = True  # Можно торговать, но с меньшим размером

        if self.current_regime in [MarketRegime.LOW_VOLATILITY]:
            reasons.append("Low volatility - expect small moves")
            should_trade = True  # Можно торговать, но с меньшими целями

        # Проверяем уверенность
        if self.regime_confidence < 0.4:
            reasons.append("Low confidence in regime detection")
            should_trade = False

        # Проверяем экстремальные условия
        if self.metrics.get('trend', {}).get('adx', 0) > 50:
            reasons.append("Extremely strong trend - caution with reversals")
            # В сильном тренде торгуем только по тренду

        return {
            'should_trade': should_trade,
            'reasons': reasons,
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence
        }

    def get_trade_bias(self) -> int:
        """
        Возвращает предпочтительное направление торговли в текущем режиме
        1 = только LONG, -1 = только SHORT, 0 = оба направления
        """
        if self.current_regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            return 1
        elif self.current_regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
            return -1
        else:
            return 0