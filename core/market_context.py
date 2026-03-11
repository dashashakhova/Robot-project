"""
Market context analysis for rule-based trading strategies.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    """High-level market regimes."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class MarketContextAnalyzer:
    """
    Analyzes market structure (trend, volatility, levels, liquidity)
    and provides simple trade filters.
    """

    def __init__(
        self,
        trend_fast_period: int = 20,
        trend_slow_period: int = 50,
        atr_period: int = 14,
        sr_window: int = 120,
        volume_window: int = 20,
        lookback_window: int = 240,
        high_volatility_ratio: float = 1.8,
        low_volatility_ratio: float = 0.7,
        block_volatility_ratio: float = 3.0,
        min_volume_ratio: float = 0.5,
        min_trend_strength_pct: float = 0.15,
    ):
        self.trend_fast_period = trend_fast_period
        self.trend_slow_period = trend_slow_period
        self.atr_period = atr_period
        self.sr_window = sr_window
        self.volume_window = volume_window
        self.lookback_window = lookback_window
        self.high_volatility_ratio = high_volatility_ratio
        self.low_volatility_ratio = low_volatility_ratio
        self.block_volatility_ratio = block_volatility_ratio
        self.min_volume_ratio = min_volume_ratio
        self.min_trend_strength_pct = min_trend_strength_pct

        self.last_context: Optional[Dict[str, Any]] = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Builds market context from OHLCV dataframe.
        """
        if df is None or df.empty:
            context = self._empty_context("empty_dataframe")
            self.last_context = context
            return context

        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            context = self._empty_context(f"missing_columns:{','.join(sorted(missing_columns))}")
            self.last_context = context
            return context

        data = df.tail(max(self.lookback_window, self.trend_slow_period + 10)).copy()

        current_price = self._to_float(data["close"].iloc[-1])
        current_volume = self._to_float(data["volume"].iloc[-1])
        timestamp = data.index[-1]

        # Trend block
        fast_ma = data["close"].rolling(self.trend_fast_period).mean()
        slow_ma = data["close"].rolling(self.trend_slow_period).mean()
        fast_value = self._to_float(fast_ma.iloc[-1], default=current_price)
        slow_value = self._to_float(slow_ma.iloc[-1], default=current_price)

        trend_strength_pct = 0.0
        if current_price > 0:
            trend_strength_pct = ((fast_value - slow_value) / current_price) * 100.0

        slope_lookback = min(10, len(data) - 1)
        slope_pct = 0.0
        if slope_lookback > 0 and len(slow_ma.dropna()) > slope_lookback:
            prev_slow = self._to_float(slow_ma.iloc[-(slope_lookback + 1)], default=slow_value)
            if current_price > 0:
                slope_pct = ((slow_value - prev_slow) / current_price) * 100.0

        if trend_strength_pct > 0.05:
            trend_direction = 1
        elif trend_strength_pct < -0.05:
            trend_direction = -1
        else:
            trend_direction = 0

        is_trending = (
            abs(trend_strength_pct) >= self.min_trend_strength_pct
            or abs(slope_pct) >= self.min_trend_strength_pct
        )

        # Volatility block
        atr_series = self._calculate_atr(data, period=self.atr_period)
        atr_value = self._to_float(atr_series.iloc[-1], default=0.0)
        atr_pct = (atr_value / current_price * 100.0) if current_price > 0 else 0.0

        atr_pct_series = (
            atr_series / data["close"].replace(0, np.nan) * 100.0
        ).replace([np.inf, -np.inf], np.nan)
        baseline_atr_pct = self._to_float(
            atr_pct_series.tail(self.lookback_window).median(), default=atr_pct
        )
        if baseline_atr_pct <= 0:
            baseline_atr_pct = max(atr_pct, 1e-9)

        volatility_ratio = atr_pct / baseline_atr_pct if baseline_atr_pct > 0 else 1.0
        is_high_volatility = volatility_ratio >= self.high_volatility_ratio
        is_low_volatility = volatility_ratio <= self.low_volatility_ratio

        # Support/resistance block
        sr_slice = data.tail(min(self.sr_window, len(data)))
        recent_high = self._to_float(sr_slice["high"].max(), default=current_price)
        recent_low = self._to_float(sr_slice["low"].min(), default=current_price)

        nearest_resistance = recent_high if recent_high > current_price else None
        nearest_support = recent_low if recent_low < current_price else None

        resistance_distance_pct = (
            ((nearest_resistance - current_price) / current_price) * 100.0
            if nearest_resistance is not None and current_price > 0
            else None
        )
        support_distance_pct = (
            ((current_price - nearest_support) / current_price) * 100.0
            if nearest_support is not None and current_price > 0
            else None
        )

        # Liquidity block
        avg_volume = self._to_float(
            data["volume"].tail(self.volume_window).mean(), default=current_volume
        )
        volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        is_thin = volume_ratio < self.min_volume_ratio

        regime = self._classify_regime(
            trend_direction=trend_direction,
            trend_strength_pct=trend_strength_pct,
            is_trending=is_trending,
            is_high_volatility=is_high_volatility,
            is_low_volatility=is_low_volatility,
        )
        regime_confidence = self._regime_confidence(
            regime=regime,
            trend_strength_pct=trend_strength_pct,
            volatility_ratio=volatility_ratio,
        )

        context = {
            "timestamp": str(timestamp),
            "regime": regime.value,
            "regime_confidence": regime_confidence,
            "price": {
                "current": current_price,
            },
            "trend": {
                "direction": trend_direction,
                "strength_pct": trend_strength_pct,
                "slope_pct": slope_pct,
                "fast_ma": fast_value,
                "slow_ma": slow_value,
                "is_trending": is_trending,
            },
            "volatility": {
                "atr": atr_value,
                "current_atr_pct": atr_pct,
                "baseline_atr_pct": baseline_atr_pct,
                "volatility_ratio": volatility_ratio,
                "is_high_volatility": is_high_volatility,
                "is_low_volatility": is_low_volatility,
            },
            "support_resistance": {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_distance_pct": support_distance_pct,
                "resistance_distance_pct": resistance_distance_pct,
                "window_bars": len(sr_slice),
            },
            "liquidity": {
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "is_thin": is_thin,
            },
        }

        self.last_context = context
        return context

    def should_trade(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns trade permission and reasons for blocking.
        """
        context = context or self.last_context
        if context is None:
            return {"should_trade": False, "reasons": ["no_context"]}

        reasons = []

        regime_value = context.get("regime", MarketRegime.UNKNOWN.value)
        if regime_value == MarketRegime.UNKNOWN.value:
            reasons.append("unknown_regime")

        volatility = context.get("volatility", {})
        if self._to_float(volatility.get("volatility_ratio"), default=1.0) >= self.block_volatility_ratio:
            reasons.append("extreme_volatility")

        liquidity = context.get("liquidity", {})
        if bool(liquidity.get("is_thin", False)):
            reasons.append("low_liquidity")

        return {"should_trade": len(reasons) == 0, "reasons": reasons}

    def get_trade_bias(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Returns directional market bias:
        1 = long bias, -1 = short bias, 0 = neutral.
        """
        context = context or self.last_context
        if context is None:
            return 0

        regime_value = context.get("regime", MarketRegime.UNKNOWN.value)
        trend = context.get("trend", {})
        trend_direction = int(trend.get("direction", 0))
        trend_strength_pct = abs(self._to_float(trend.get("strength_pct"), default=0.0))

        if regime_value == MarketRegime.TRENDING_UP.value:
            return 1
        if regime_value == MarketRegime.TRENDING_DOWN.value:
            return -1

        if regime_value == MarketRegime.HIGH_VOLATILITY.value and trend_strength_pct >= self.min_trend_strength_pct:
            return trend_direction

        return 0

    def _classify_regime(
        self,
        trend_direction: int,
        trend_strength_pct: float,
        is_trending: bool,
        is_high_volatility: bool,
        is_low_volatility: bool,
    ) -> MarketRegime:
        if is_high_volatility:
            return MarketRegime.HIGH_VOLATILITY

        if is_trending and trend_direction > 0 and abs(trend_strength_pct) >= self.min_trend_strength_pct:
            return MarketRegime.TRENDING_UP
        if is_trending and trend_direction < 0 and abs(trend_strength_pct) >= self.min_trend_strength_pct:
            return MarketRegime.TRENDING_DOWN

        if is_low_volatility:
            return MarketRegime.LOW_VOLATILITY

        return MarketRegime.RANGING

    @staticmethod
    def _regime_confidence(
        regime: MarketRegime,
        trend_strength_pct: float,
        volatility_ratio: float,
    ) -> float:
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return float(np.clip(abs(trend_strength_pct) / 0.6, 0.0, 1.0))
        if regime == MarketRegime.HIGH_VOLATILITY:
            return float(np.clip((volatility_ratio - 1.0) / 2.0, 0.0, 1.0))
        if regime == MarketRegime.LOW_VOLATILITY:
            return float(np.clip((1.0 - volatility_ratio) / 0.6, 0.0, 1.0))
        if regime == MarketRegime.RANGING:
            return 0.6
        return 0.0

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _empty_context(reason: str) -> Dict[str, Any]:
        return {
            "timestamp": None,
            "regime": MarketRegime.UNKNOWN.value,
            "regime_confidence": 0.0,
            "reason": reason,
            "price": {"current": 0.0},
            "trend": {
                "direction": 0,
                "strength_pct": 0.0,
                "slope_pct": 0.0,
                "fast_ma": 0.0,
                "slow_ma": 0.0,
                "is_trending": False,
            },
            "volatility": {
                "atr": 0.0,
                "current_atr_pct": 0.0,
                "baseline_atr_pct": 0.0,
                "volatility_ratio": 1.0,
                "is_high_volatility": False,
                "is_low_volatility": False,
            },
            "support_resistance": {
                "nearest_support": None,
                "nearest_resistance": None,
                "support_distance_pct": None,
                "resistance_distance_pct": None,
                "window_bars": 0,
            },
            "liquidity": {
                "current_volume": 0.0,
                "avg_volume": 0.0,
                "volume_ratio": 1.0,
                "is_thin": False,
            },
        }
