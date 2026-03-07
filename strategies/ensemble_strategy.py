"""
Ансамблевая стратегия, комбинирующая несколько моделей
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from collections import defaultdict

from strategies.base_strategy import BaseStrategy
from strategies.ml_strategy import MLStrategy
from core.config import config


class EnsembleStrategy(BaseStrategy):
    """
    Ансамблевая стратегия, комбинирующая несколько моделей для повышения надежности
    """

    def __init__(self, instrument: str):
        super().__init__("Ensemble", instrument)

        # Инициализируем отдельные стратегии
        self.strategies = {
            'ml_rf': MLStrategy(instrument),  # Random Forest
            # Здесь можно добавить другие стратегии
        }

        # Веса стратегий (будут обновляться на основе производительности)
        self.weights = {name: 1.0 for name in self.strategies.keys()}

        # История производительности для динамических весов
        self.performance_history = defaultdict(list)

        # Параметры ансамбля
        self.min_agreement = 0.5  # Минимальное согласие стратегий
        self.weight_decay = 0.95  # Коэффициент затухания для старых результатов
        self.performance_window = 20  # Окно для оценки производительности

        # Метрики
        self.strategy_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает признаки (делегирует ML стратегии)"""
        return self.strategies['ml_rf'].prepare_features(df)

    def train(self, df: pd.DataFrame, optimize: bool = False) -> Dict:
        """
        Обучает все стратегии в ансамбле

        Args:
            df: DataFrame с данными
            optimize: выполнить оптимизацию

        Returns:
            Метрики обучения
        """
        print(f"\n🎯 Обучение ансамбля стратегий для {self.instrument}...")

        results = {}

        for name, strategy in self.strategies.items():
            print(f"\n   Обучаем {name}...")
            try:
                metrics = strategy.train(df, optimize)
                results[name] = metrics

                # Сохраняем модель
                strategy.save_model()

            except Exception as e:
                print(f"   ❌ Ошибка обучения {name}: {e}")
                results[name] = {'error': str(e)}

        # Инициализируем веса поровну
        n_successful = sum(1 for r in results.values() if 'error' not in r)
        if n_successful > 0:
            for name in self.strategies.keys():
                if 'error' not in results.get(name, {}):
                    self.weights[name] = 1.0 / n_successful

        self.is_trained = True
        self.last_update = datetime.now()
        self.strategy_metrics = results

        print(f"\n✅ Ансамбль обучен. Активных стратегий: {n_successful}")

        return results

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Комбинирует предсказания всех стратегий

        Args:
            df: DataFrame с данными

        Returns:
            Ансамблевый прогноз
        """
        if not self.is_trained:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Ensemble not trained']
            }

        predictions = []
        confidences = []
        reasons = []

        # Собираем предсказания от всех стратегий
        for name, strategy in self.strategies.items():
            try:
                pred = strategy.predict(df)

                if pred['action'] != 'HOLD':
                    predictions.append({
                        'strategy': name,
                        'action': pred['action'],
                        'confidence': pred['confidence'],
                        'weight': self.weights.get(name, 1.0)
                    })
                    confidences.append(pred['confidence'])
                    reasons.append(f"{name}: {pred['action']} ({pred['confidence']:.1%})")
            except Exception as e:
                print(f"   ⚠️ Ошибка предсказания {name}: {e}")

        if not predictions:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': ['No strategy generated signal']
            }

        # Голосование с учетом весов
        buy_weight = sum(p['weight'] for p in predictions if p['action'] == 'BUY')
        sell_weight = sum(p['weight'] for p in predictions if p['action'] == 'SELL')
        total_weight = sum(p['weight'] for p in predictions)

        # Определяем действие
        if buy_weight > sell_weight:
            action = 'BUY'
            agreement = buy_weight / total_weight
            avg_confidence = np.mean([p['confidence'] for p in predictions if p['action'] == 'BUY'])
        elif sell_weight > buy_weight:
            action = 'SELL'
            agreement = sell_weight / total_weight
            avg_confidence = np.mean([p['confidence'] for p in predictions if p['action'] == 'SELL'])
        else:
            action = 'HOLD'
            agreement = 0.5
            avg_confidence = 0.0

        # Финальная уверенность = согласие * средняя уверенность
        confidence = agreement * avg_confidence if avg_confidence > 0 else 0

        # Проверяем минимальное согласие
        if agreement < self.min_agreement:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasons': [f'Insufficient agreement: {agreement:.1%}']
            }

        return {
            'action': action,
            'confidence': float(confidence),
            'agreement': float(agreement),
            'reasons': reasons,
            'predictions': predictions,
            'current_price': float(df['close'].iloc[-1])
        }

    def update_weights(self, trade_results: List[Dict]):
        """
        Обновляет веса стратегий на основе их производительности

        Args:
            trade_results: список результатов сделок
        """
        if not trade_results:
            return

        # Анализируем производительность каждой стратегии
        strategy_performance = defaultdict(lambda: {'correct': 0, 'total': 0})

        for trade in trade_results:
            # В реальном проекте здесь нужно сопоставить сделку с предсказаниями
            # Пока используем упрощенный подход
            pass

        # Обновляем веса
        total_performance = 0
        new_weights = {}

        for name in self.strategies.keys():
            perf = strategy_performance[name]
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                new_weights[name] = accuracy
                total_performance += accuracy

        # Нормализуем веса
        if total_performance > 0:
            for name in new_weights:
                self.weights[name] = new_weights[name] / total_performance

        # Применяем затухание
        for name in self.weights:
            self.weights[name] *= self.weight_decay

    def get_strategy_stats(self) -> Dict:
        """
        Возвращает статистику по каждой стратегии

        Returns:
            Словарь со статистикой
        """
        stats = {}

        for name, strategy in self.strategies.items():
            stats[name] = {
                'weight': self.weights.get(name, 0),
                'is_trained': strategy.is_trained,
                'last_update': strategy.last_update,
                'metrics': strategy.training_metrics if hasattr(strategy, 'training_metrics') else {}
            }

        return stats

    def save_all_models(self):
        """Сохраняет все модели ансамбля"""
        for name, strategy in self.strategies.items():
            strategy.save_model(suffix=f"ensemble_{name}")

        # Сохраняем веса
        weights_file = self.models_dir / f"{self.name}_weights.joblib"
        joblib.dump({
            'weights': self.weights,
            'performance_history': self.performance_history,
            'last_update': self.last_update
        }, weights_file)

    def load_all_models(self) -> bool:
        """
        Загружает все модели ансамбля

        Returns:
            True если все модели загружены
        """
        success = True

        for name, strategy in self.strategies.items():
            if not strategy.load_model(suffix=f"ensemble_{name}"):
                success = False
                print(f"   ⚠️ Не удалось загрузить {name}")

        # Загружаем веса
        weights_file = self.models_dir / f"{self.name}_weights.joblib"
        if weights_file.exists():
            weights_data = joblib.load(weights_file)
            self.weights = weights_data['weights']
            self.performance_history = weights_data['performance_history']
            self.last_update = weights_data['last_update']

        self.is_trained = success
        return success

    def get_params(self) -> Dict:
        """Возвращает параметры стратегии"""
        return {
            'strategies': list(self.strategies.keys()),
            'weights': self.weights,
            'min_agreement': self.min_agreement,
            'weight_decay': self.weight_decay,
            'performance_window': self.performance_window
        }

    def set_params(self, params: Dict):
        """Устанавливает параметры стратегии"""
        if 'min_agreement' in params:
            self.min_agreement = params['min_agreement']
        if 'weight_decay' in params:
            self.weight_decay = params['weight_decay']
        if 'performance_window' in params:
            self.performance_window = params['performance_window']