"""
Стратегия на основе машинного обучения (Random Forest)
Альтернативная версия без XGBoost для совместимости с Mac
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from strategies.base_strategy import BaseStrategy
from core.config import config


class MLStrategy(BaseStrategy):
    """Стратегия на основе Random Forest (без XGBoost)"""

    def __init__(self, instrument: str):
        super().__init__("ML_RandomForest", instrument)

        # Модель
        self.model = None
        self.scaler = StandardScaler()

        # Параметры модели
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'  # для несбалансированных классов
        }

        # Признаки
        self.feature_columns = []

        # Метрики
        self.training_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает расширенный набор признаков для ML

        Args:
            df: DataFrame с базовыми индикаторами

        Returns:
            DataFrame с признаками
        """
        data = df.copy()

        # 1. Базовые индикаторы
        base_features = [
            'rsi', 'macd_line', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'atr_percent',
            'price_range', 'momentum'
        ]

        # Проверяем наличие колонок
        available_features = [f for f in base_features if f in data.columns]

        # 2. Скользящие средние и их отношения
        for period in [5, 10, 20, 50]:
            sma_col = f'sma_{period}'
            if sma_col in data.columns:
                # Отклонение от SMA
                data[f'price_to_sma_{period}'] = (data['close'] - data[sma_col]) / data[sma_col]

                # Наклон SMA
                data[f'sma_{period}_slope'] = data[sma_col].pct_change(5)

        # 3. Лаговые признаки
        for col in ['close', 'volume', 'rsi', 'macd_hist']:
            if col in data.columns:
                for lag in [1, 2, 3, 5, 10]:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)

        # 4. Скользящие статистики
        for window in [5, 10, 20]:
            data[f'close_rolling_mean_{window}'] = data['close'].rolling(window).mean()
            data[f'close_rolling_std_{window}'] = data['close'].rolling(window).std()
            data[f'volume_rolling_mean_{window}'] = data['volume'].rolling(window).mean()

        # 5. Признаки волатильности
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']

        # 6. Временные признаки
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek

        # 7. Циклические признаки
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

        # 8. Взаимодействия признаков (только если признаки существуют)
        if 'rsi' in data.columns and 'macd_hist' in data.columns:
            data['rsi_macd_interaction'] = data['rsi'] * data['macd_hist']

        if 'volume_ratio' in data.columns and 'momentum' in data.columns:
            data['volume_price_interaction'] = data['volume_ratio'] * data['momentum']

        # Определяем список признаков
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target_price',
                        'target_return', 'target_direction']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]

        # Удаляем строки с NaN
        data = data.dropna()

        return data

    def _create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Создает целевую переменную для классификации

        Args:
            df: DataFrame с данными
            horizon: горизонт прогноза в часах

        Returns:
            Series с классами: 1 (рост), 0 (боковик), -1 (падение)
        """
        # Будущая доходность
        future_return = (df['close'].shift(-horizon) / df['close'] - 1) * 100

        # Определяем пороги для классификации
        min_profit = config.get('strategy', 'min_profit', default=0.5)

        target = pd.Series(0, index=df.index)
        target[future_return > min_profit] = 1
        target[future_return < -min_profit] = -1

        return target

    def train(self, df: pd.DataFrame, optimize: bool = False) -> Dict:
        """
        Обучает модель Random Forest

        Args:
            df: DataFrame с данными
            optimize: выполнить оптимизацию гиперпараметров

        Returns:
            Метрики обучения
        """
        print(f"\n🎯 Обучение ML стратегии для {self.instrument}...")

        # Подготавливаем данные
        data = self.prepare_features(df)

        if len(data) < 100:
            return {'error': f'Insufficient data: {len(data)} rows'}

        # Создаем целевую переменную
        horizon = self.forecast_hours[0] if self.forecast_hours else 1
        y = self._create_target(data, horizon)

        # Убираем строки с NaN в целевой переменной
        valid_idx = ~y.isna()
        X = data.loc[valid_idx, self.feature_columns]
        y = y[valid_idx]

        if len(X) == 0:
            return {'error': 'No valid data after preprocessing'}

        # Масштабируем признаки
        X_scaled = self.scaler.fit_transform(X)

        # Разделяем на train/val с учетом временного порядка
        train_size = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

        # Оптимизация гиперпараметров
        if optimize:
            self._optimize_hyperparameters(X_train, y_train)

        # Обучаем модель
        print("   Обучаем Random Forest...")
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        # Оцениваем на валидации
        y_pred = self.model.predict(X_val)

        # Метрики для каждого класса
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        self.training_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_features': len(self.feature_columns),
            'class_distribution': y_train.value_counts().to_dict()
        }

        # Важность признаков
        self._calculate_feature_importance()

        self.is_trained = True
        self.last_update = datetime.now()

        # Сохраняем модель
        self.save_model()

        print(f"✅ Обучение завершено:")
        print(f"   Точность: {accuracy:.2%}")
        print(f"   F1-score: {f1:.2%}")
        print(f"   Обучено на {len(X_train)} примерах")

        return self.training_metrics

    def _optimize_hyperparameters(self, X, y):
        """Оптимизирует гиперпараметры с помощью GridSearchCV"""
        print("   Оптимизация гиперпараметров...")

        # TimeSeriesSplit для корректной валидации
        tscv = TimeSeriesSplit(n_splits=3)

        # Сетка параметров
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [10, 20, 50],
            'min_samples_leaf': [5, 10, 20]
        }

        # GridSearch
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)

        # Обновляем параметры
        self.model_params.update(grid_search.best_params_)
        print(f"   Лучшие параметры: {grid_search.best_params_}")
        print(f"   Лучший F1-score: {grid_search.best_score_:.4f}")

    def _calculate_feature_importance(self):
        """Рассчитывает важность признаков"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_

            self.feature_importance = {
                self.feature_columns[i]: float(importance[i])
                for i in range(len(self.feature_columns))
            }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Делает прогноз на основе текущих данных

        Args:
            df: DataFrame с последними данными

        Returns:
            Словарь с прогнозом
        """
        if not self.is_trained or self.model is None:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Model not trained']
            }

        try:
            # Подготавливаем признаки
            data = self.prepare_features(df)

            if len(data) == 0:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasons': ['Insufficient data for features']
                }

            # Берем последнюю строку
            last_row = data.iloc[-1:][self.feature_columns]

            # Проверяем на наличие NaN
            if last_row.isnull().any().any():
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasons': ['NaN values in features']
                }

            # Масштабируем
            X = self.scaler.transform(last_row)

            # Получаем предсказание
            prediction = self.model.predict(X)[0]

            # Получаем вероятности
            probabilities = self.model.predict_proba(X)[0]

            # Определяем действие
            if prediction == 1:
                action = 'BUY'
                confidence = probabilities[self.model.classes_.tolist().index(1)] if 1 in self.model.classes_ else 0.5
            elif prediction == -1:
                action = 'SELL'
                confidence = probabilities[self.model.classes_.tolist().index(-1)] if -1 in self.model.classes_ else 0.5
            else:
                action = 'HOLD'
                confidence = 0.0

            # Дополнительная информация
            proba_info = {}
            for i, cls in enumerate(self.model.classes_):
                proba_info[f'proba_{cls}'] = float(probabilities[i])

            return {
                'action': action,
                'confidence': float(confidence),
                'reasons': ['ML prediction'],
                'prediction': int(prediction),
                'current_price': float(df['close'].iloc[-1]),
                **proba_info
            }

        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': [f'Prediction error: {str(e)}']
            }

    def get_params(self) -> Dict:
        """Возвращает параметры стратегии"""
        return {
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'forecast_hours': self.forecast_hours,
            'min_confidence': self.min_confidence,
            'training_metrics': self.training_metrics
        }

    def set_params(self, params: Dict):
        """Устанавливает параметры стратегии"""
        if 'model_params' in params:
            self.model_params.update(params['model_params'])
        if 'forecast_hours' in params:
            self.forecast_hours = params['forecast_hours']
        if 'min_confidence' in params:
            self.min_confidence = params['min_confidence']