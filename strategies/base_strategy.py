"""
Базовый класс для всех торговых стратегий
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

from core.config import config
from core.market_context import MarketContextAnalyzer, MarketRegime


class BaseStrategy(ABC):
    """Абстрактный базовый класс для всех стратегий"""

    def __init__(self, name: str, instrument: str):
        """
        Args:
            name: название стратегии
            instrument: тикер инструмента
        """
        self.name = name
        self.instrument = instrument
        self.market_analyzer = MarketContextAnalyzer()
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        self.last_update = None

        # Параметры из конфига
        self.min_confidence = config.get('strategy', 'min_confidence', default=0.6)
        self.forecast_hours = config.get('strategy', 'forecast_hours', default=[1, 2, 4])

        # Директория для сохранения моделей
        self.models_dir = Path("models") / instrument
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает признаки для модели

        Args:
            df: DataFrame с сырыми данными и индикаторами

        Returns:
            DataFrame с признаками
        """
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Обучает стратегию на исторических данных

        Args:
            df: DataFrame с данными

        Returns:
            Словарь с метриками обучения
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Делает прогноз на текущий момент

        Args:
            df: DataFrame с последними данными

        Returns:
            Словарь с прогнозом
        """
        pass

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Получает торговый сигнал с учетом рыночного контекста

        Args:
            df: DataFrame с данными

        Returns:
            Словарь с сигналом
        """
        if not self.is_trained:
            return {
                'action': 'HOLD',
                'reason': 'Strategy not trained',
                'confidence': 0.0
            }

        # Анализируем рыночный контекст
        market_context = self.market_analyzer.analyze(df)

        # Проверяем, можно ли торговать
        trade_check = self.market_analyzer.should_trade()
        if not trade_check['should_trade']:
            return {
                'action': 'HOLD',
                'reason': f"Market conditions unfavorable: {trade_check['reasons']}",
                'confidence': 0.0,
                'market_context': market_context
            }

        # Получаем базовый прогноз
        prediction = self.predict(df)

        # Корректируем с учетом рыночного контекста
        signal = self._adjust_signal(prediction, market_context)

        # Добавляем метаданные
        signal['market_context'] = market_context
        signal['strategy'] = self.name
        signal['instrument'] = self.instrument
        signal['timestamp'] = datetime.now()

        return signal

    def _adjust_signal(self, prediction: Dict, market_context: Dict) -> Dict:
        """
        Корректирует сигнал с учетом рыночного контекста

        Args:
            prediction: базовый прогноз
            market_context: результаты анализа рынка

        Returns:
            Скорректированный сигнал
        """
        action = prediction.get('action', 'HOLD')
        confidence = prediction.get('confidence', 0.0)
        reasons = prediction.get('reasons', [])

        # Получаем предпочтительное направление торговли
        bias = self.market_analyzer.get_trade_bias()

        # Корректируем сигнал в соответствии с рыночным режимом
        if bias == 1 and action == 'SELL':
            # Не продаем в сильном восходящем тренде
            action = 'HOLD'
            reasons.append("Strong uptrend - avoiding shorts")
            confidence *= 0.5
        elif bias == -1 and action == 'BUY':
            # Не покупаем в сильном нисходящем тренде
            action = 'HOLD'
            reasons.append("Strong downtrend - avoiding longs")
            confidence *= 0.5

        # Корректируем уверенность на основе волатильности
        volatility = market_context.get('volatility', {})
        if volatility.get('is_high_volatility', False):
            confidence *= 0.8  # Уменьшаем уверенность при высокой волатильности
            reasons.append("High volatility - reduced confidence")

        # Проверяем уровни поддержки/сопротивления
        sr = market_context.get('support_resistance', {})
        current_price = df['close'].iloc[-1] if 'df' in locals() else prediction.get('current_price')

        if action == 'BUY' and sr.get('nearest_resistance'):
            dist_to_resistance = sr['resistance_distance_pct']
            if dist_to_resistance and dist_to_resistance < 1.0:
                # Слишком близко к сопротивлению
                confidence *= 0.7
                reasons.append(f"Close to resistance ({dist_to_resistance:.2f}%)")

        elif action == 'SELL' and sr.get('nearest_support'):
            dist_to_support = sr['support_distance_pct']
            if dist_to_support and dist_to_support < 1.0:
                # Слишком близко к поддержке
                confidence *= 0.7
                reasons.append(f"Close to support ({dist_to_support:.2f}%)")

        return {
            'action': action,
            'confidence': min(confidence, 1.0),
            'reasons': reasons,
            'original_prediction': prediction
        }

    def save_model(self, suffix: str = ""):
        """Сохраняет обученную модель"""
        if suffix:
            filename = self.models_dir / f"{self.name}_{suffix}.joblib"
        else:
            filename = self.models_dir / f"{self.name}_latest.joblib"

        # Сохраняем состояние модели
        model_state = {
            'name': self.name,
            'instrument': self.instrument,
            'is_trained': self.is_trained,
            'training_history': self.training_history[-100:],  # последние 100 записей
            'feature_importance': self.feature_importance,
            'last_update': self.last_update,
            'params': self.get_params()
        }

        joblib.dump(model_state, filename)
        return filename

    def load_model(self, suffix: str = ""):
        """Загружает обученную модель"""
        if suffix:
            filename = self.models_dir / f"{self.name}_{suffix}.joblib"
        else:
            filename = self.models_dir / f"{self.name}_latest.joblib"

        if not filename.exists():
            return False

        model_state = joblib.load(filename)
        self.name = model_state['name']
        self.instrument = model_state['instrument']
        self.is_trained = model_state['is_trained']
        self.training_history = model_state['training_history']
        self.feature_importance = model_state['feature_importance']
        self.last_update = model_state['last_update']
        self.set_params(model_state['params'])

        return True

    @abstractmethod
    def get_params(self) -> Dict:
        """Возвращает параметры стратегии"""
        pass

    @abstractmethod
    def set_params(self, params: Dict):
        """Устанавливает параметры стратегии"""
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        """Возвращает важность признаков в виде DataFrame"""
        if not self.feature_importance:
            return pd.DataFrame()

        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)

        return df

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Валидирует стратегию на отложенных данных

        Args:
            df: DataFrame с данными для валидации

        Returns:
            Метрики валидации
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}

        predictions = []
        actuals = []

        for i in range(len(df) - max(self.forecast_hours)):
            row = df.iloc[i:i + 1]
            pred = self.predict(row)

            if pred['action'] != 'HOLD':
                future_return = (df['close'].iloc[i + self.forecast_hours[0]] / row['close'].iloc[-1] - 1) * 100

                predictions.append({
                    'time': row.index[0],
                    'action': pred['action'],
                    'confidence': pred['confidence']
                })
                actuals.append(future_return)

        if not predictions:
            return {'error': 'No predictions made'}

        # Рассчитываем метрики
        correct = 0
        for pred, actual in zip(predictions, actuals):
            if pred['action'] == 'BUY' and actual > 0:
                correct += 1
            elif pred['action'] == 'SELL' and actual < 0:
                correct += 1

        accuracy = correct / len(predictions) if predictions else 0

        return {
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'avg_confidence': np.mean([p['confidence'] for p in predictions]),
            'validation_period': f"{df.index[0]} - {df.index[-1]}"
        }