"""
Base class for trading strategies used in this project.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from core.config import config
from core.market_context import MarketContextAnalyzer


class BaseStrategy(ABC):
    """
    Lightweight base strategy for rule-based systems.

    Subclasses can either:
    - implement `generate_signal(df)` and use inherited `get_signal`, or
    - implement `get_signal(df)` directly (it will be auto-wrapped and post-processed).
    """

    _WRAPPED_ATTR = "_base_strategy_wrapped"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        custom_get_signal = cls.__dict__.get("get_signal")
        if custom_get_signal is None:
            return
        if getattr(custom_get_signal, BaseStrategy._WRAPPED_ATTR, False):
            return

        @wraps(custom_get_signal)
        def wrapped_get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
            raw_signal = custom_get_signal(self, df)
            return self.post_process_signal(raw_signal, df)

        setattr(wrapped_get_signal, BaseStrategy._WRAPPED_ATTR, True)
        setattr(cls, "_raw_get_signal", custom_get_signal)
        setattr(cls, "get_signal", wrapped_get_signal)

    def __init__(self, name: str, instrument: str):
        self.name = name
        self.instrument = instrument
        self.market_analyzer = MarketContextAnalyzer()
        self.last_update: Optional[datetime] = None

        # Strategy/runtime settings from config
        self.min_confidence = config.get("strategy", "min_confidence", default=0.6)
        self.use_market_filters = True

        # Optional metadata used by save/load methods
        self.training_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, float] = {}
        self.is_trained = True  # Rule-based strategies are usable immediately

        self.models_dir = Path("models") / instrument
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares features. For rule-based strategies this can return df unchanged."""

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optional signal generator for subclasses that do not override `get_signal`.
        """
        return {"signal": "HOLD", "confidence": 0.0, "reason": ["No signal logic"]}

    def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Default signal pipeline.
        """
        raw_signal = self.generate_signal(df)
        return self.post_process_signal(raw_signal, df)

    def post_process_signal(self, raw_signal: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Normalizes signal format and applies context-based filters.
        """
        signal = self._normalize_signal(raw_signal or {}, df)

        market_context = self.market_analyzer.analyze(df) if self.use_market_filters else None
        if market_context is not None:
            signal["market_context"] = market_context

            if signal["signal"] != "HOLD":
                signal = self._adjust_with_market_context(signal, market_context)

                trade_check = self.market_analyzer.should_trade(market_context)
                if not trade_check["should_trade"]:
                    signal = self._to_hold(
                        signal,
                        [f"Market filter: {', '.join(trade_check['reasons'])}"],
                    )

        if signal["signal"] != "HOLD" and signal["confidence"] < self.min_confidence:
            signal = self._to_hold(
                signal,
                [f"Confidence below threshold {self.min_confidence:.2f}"],
            )

        return self._finalize_signal(signal, df)

    def _normalize_signal(self, raw_signal: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        signal = dict(raw_signal)

        action = str(signal.get("signal", signal.get("action", "HOLD"))).upper()
        if action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"

        reasons = signal.get("reason", signal.get("reasons", []))
        if isinstance(reasons, str):
            reasons = [reasons]
        elif reasons is None:
            reasons = []
        elif not isinstance(reasons, list):
            reasons = [str(reasons)]

        clean_reasons = [str(item) for item in reasons if str(item).strip()]

        confidence = self._clamp(self._safe_float(signal.get("confidence"), default=0.0), 0.0, 1.0)
        current_price = self._extract_current_price(df, signal)

        signal["signal"] = action
        signal["action"] = action
        signal["confidence"] = confidence
        signal["reason"] = clean_reasons
        signal["reasons"] = list(clean_reasons)

        if current_price is not None:
            signal.setdefault("current_price", current_price)
            signal.setdefault("entry_price", current_price)

        if action == "HOLD":
            signal.setdefault("stop_loss", None)
            signal.setdefault("take_profit", None)

        return signal

    def _adjust_with_market_context(
        self,
        signal: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = signal["signal"]
        confidence = signal["confidence"]
        reasons = list(signal["reason"])

        bias = self.market_analyzer.get_trade_bias(market_context)
        if bias == 1 and action == "SELL":
            return self._to_hold(signal, ["Bullish regime blocks shorts"])
        if bias == -1 and action == "BUY":
            return self._to_hold(signal, ["Bearish regime blocks longs"])

        volatility = market_context.get("volatility", {})
        if bool(volatility.get("is_high_volatility", False)):
            confidence *= 0.8
            reasons.append("High volatility: confidence reduced")

        liquidity = market_context.get("liquidity", {})
        if bool(liquidity.get("is_thin", False)):
            confidence *= 0.7
            reasons.append("Low liquidity: confidence reduced")

        sr = market_context.get("support_resistance", {})
        if action == "BUY":
            dist = sr.get("resistance_distance_pct")
            if dist is not None and float(dist) < 0.4:
                confidence *= 0.75
                reasons.append(f"Close to resistance ({float(dist):.2f}%)")
        elif action == "SELL":
            dist = sr.get("support_distance_pct")
            if dist is not None and float(dist) < 0.4:
                confidence *= 0.75
                reasons.append(f"Close to support ({float(dist):.2f}%)")

        signal["confidence"] = self._clamp(confidence, 0.0, 1.0)
        signal["reason"] = reasons
        signal["reasons"] = list(reasons)
        return signal

    def _to_hold(self, signal: Dict[str, Any], extra_reasons: List[str]) -> Dict[str, Any]:
        reasons = list(signal.get("reason", [])) + [r for r in extra_reasons if r]
        signal["original_signal"] = signal.get("original_signal", dict(signal))
        signal["signal"] = "HOLD"
        signal["action"] = "HOLD"
        signal["confidence"] = 0.0
        signal["reason"] = reasons
        signal["reasons"] = list(reasons)
        signal["stop_loss"] = None
        signal["take_profit"] = None
        return signal

    def _finalize_signal(self, signal: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        if "timestamp" not in signal:
            signal["timestamp"] = df.index[-1] if df is not None and len(df) > 0 else datetime.now()

        signal["strategy"] = self.name
        signal["instrument"] = self.instrument
        signal["action"] = signal.get("signal", "HOLD")
        signal["signal"] = signal.get("action", "HOLD")
        signal["reason"] = list(signal.get("reason", signal.get("reasons", [])))
        signal["reasons"] = list(signal["reason"])
        signal["confidence"] = self._clamp(self._safe_float(signal.get("confidence"), default=0.0), 0.0, 1.0)
        self.last_update = datetime.now()
        return signal

    @staticmethod
    def _extract_current_price(df: pd.DataFrame, signal: Dict[str, Any]) -> Optional[float]:
        if "current_price" in signal and signal["current_price"] is not None:
            return float(signal["current_price"])
        if df is None or df.empty or "close" not in df.columns:
            return None
        return float(df["close"].iloc[-1])

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            value = float(value)
            if pd.isna(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def save_model(self, suffix: str = "") -> Path:
        """Saves strategy params and metadata."""
        filename = (
            self.models_dir / f"{self.name}_{suffix}.joblib"
            if suffix
            else self.models_dir / f"{self.name}_latest.joblib"
        )

        model_state = {
            "name": self.name,
            "instrument": self.instrument,
            "is_trained": self.is_trained,
            "training_history": self.training_history[-100:],
            "feature_importance": self.feature_importance,
            "last_update": self.last_update,
            "min_confidence": self.min_confidence,
            "use_market_filters": self.use_market_filters,
            "params": self.get_params(),
        }
        joblib.dump(model_state, filename)
        return filename

    def load_model(self, suffix: str = "") -> bool:
        """Loads strategy params and metadata."""
        filename = (
            self.models_dir / f"{self.name}_{suffix}.joblib"
            if suffix
            else self.models_dir / f"{self.name}_latest.joblib"
        )
        if not filename.exists():
            return False

        model_state = joblib.load(filename)
        self.name = model_state.get("name", self.name)
        self.instrument = model_state.get("instrument", self.instrument)
        self.is_trained = model_state.get("is_trained", self.is_trained)
        self.training_history = model_state.get("training_history", self.training_history)
        self.feature_importance = model_state.get("feature_importance", self.feature_importance)
        self.last_update = model_state.get("last_update", self.last_update)
        self.min_confidence = model_state.get("min_confidence", self.min_confidence)
        self.use_market_filters = model_state.get("use_market_filters", self.use_market_filters)

        params = model_state.get("params", {})
        if isinstance(params, dict):
            self.set_params(params)
        return True

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Returns strategy parameters."""

    @abstractmethod
    def set_params(self, params: Dict[str, Any]):
        """Applies strategy parameters."""

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.feature_importance:
            return pd.DataFrame()

        return pd.DataFrame(
            [{"feature": key, "importance": value} for key, value in self.feature_importance.items()]
        ).sort_values("importance", ascending=False)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Quick validation pass over historical data.
        """
        if df is None or df.empty or "close" not in df.columns:
            return {"error": "Invalid or empty dataframe"}

        warmup = min(100, max(20, len(df) // 10))
        if len(df) <= warmup:
            return {"error": "Not enough data for validation"}

        signals = []
        for i in range(warmup, len(df)):
            sub_df = df.iloc[: i + 1]
            signal = self.get_signal(sub_df)
            signals.append(signal.get("signal", "HOLD"))

        total = len(signals)
        buys = sum(1 for s in signals if s == "BUY")
        sells = sum(1 for s in signals if s == "SELL")
        holds = total - buys - sells

        return {
            "total_samples": total,
            "buy_signals": buys,
            "sell_signals": sells,
            "hold_signals": holds,
            "actionable_ratio": (buys + sells) / total if total else 0.0,
            "validation_period": f"{df.index[0]} - {df.index[-1]}",
        }
