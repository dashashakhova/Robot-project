"""
Модуль для отслеживания качества сигналов в реальном времени
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
from pathlib import Path


class SignalQualityAnalyzer:
    """
    Анализирует качество сигналов в реальном времени и выявляет деградацию стратегии
    """

    def __init__(self, strategy_name: str, window_sizes: List[int] = [20, 50, 100]):
        """
        Args:
            strategy_name: название стратегии
            window_sizes: размеры окон для скользящих метрик
        """
        self.strategy_name = strategy_name
        self.window_sizes = window_sizes

        # История сигналов и результатов
        self.signals = deque(maxlen=1000)  # [(timestamp, signal, confidence, actual_return)]

        # Метрики по окнам
        self.metrics = {
            window: {
                'accuracy': 0.0,
                'precision': {'BUY': 0.0, 'SELL': 0.0},
                'recall': {'BUY': 0.0, 'SELL': 0.0},
                'avg_confidence': 0.0,
                'profit_factor': 0.0,
                'total_signals': 0,
                'profitable_signals': 0
            }
            for window in window_sizes
        }

        # Текущее состояние
        self.current_metrics = self.metrics[window_sizes[0]]
        self.alert_thresholds = {
            'accuracy_drop': 0.2,  # Падение точности на 20%
            'confidence_drop': 0.3,  # Падение уверенности на 30%
            'min_accuracy': 0.4,  # Минимальная допустимая точность
            'min_signals': 10  # Минимум сигналов для анализа
        }

        # История метрик для трендов
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))

        # Файл для сохранения
        self.history_file = Path(f"signal_quality_{strategy_name}.json")
        self.load_history()

    def add_signal_result(self, signal_time: datetime, signal: str,
                          confidence: float, actual_return: float):
        """
        Добавляет результат сигнала

        Args:
            signal_time: время сигнала
            signal: 'BUY' или 'SELL'
            confidence: уверенность в сигнале
            actual_return: фактическая доходность
        """
        self.signals.append({
            'timestamp': signal_time,
            'signal': signal,
            'confidence': confidence,
            'return': actual_return,
            'profitable': (signal == 'BUY' and actual_return > 0) or
                          (signal == 'SELL' and actual_return < 0)
        })

        # Обновляем метрики для всех окон
        self._update_metrics()

        # Сохраняем историю
        self.save_history()

    def _update_metrics(self):
        """Обновляет метрики для всех окон"""
        signals_list = list(self.signals)

        if len(signals_list) < min(self.window_sizes):
            return

        for window in self.window_sizes:
            if len(signals_list) >= window:
                recent = signals_list[-window:]
                self.metrics[window] = self._calculate_metrics(recent)

        # Обновляем текущие метрики (наименьшее окно)
        self.current_metrics = self.metrics[min(self.window_sizes)]

        # Сохраняем историю
        for window in self.window_sizes:
            self.metrics_history[f"accuracy_{window}"].append(
                (datetime.now(), self.metrics[window]['accuracy'])
            )
            self.metrics_history[f"confidence_{window}"].append(
                (datetime.now(), self.metrics[window]['avg_confidence'])
            )

    def _calculate_metrics(self, signals: List[Dict]) -> Dict:
        """
        Рассчитывает метрики для списка сигналов

        Args:
            signals: список сигналов

        Returns:
            Словарь с метриками
        """
        if not signals:
            return self._empty_metrics()

        df = pd.DataFrame(signals)

        # Общие метрики
        total = len(df)
        profitable = df['profitable'].sum()

        # Метрики по типам сигналов
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']

        # Точность (precision) для каждого типа
        buy_precision = buy_signals['profitable'].mean() if len(buy_signals) > 0 else 0
        sell_precision = sell_signals['profitable'].mean() if len(sell_signals) > 0 else 0

        # Полнота (recall) - доля правильных сигналов от всех возможных
        # В упрощенном виде считаем как accuracy для каждого типа
        buy_recall = buy_precision
        sell_recall = sell_precision

        # Profit factor
        buy_profit = buy_signals[buy_signals['profitable']]['return'].sum() if len(buy_signals) > 0 else 0
        buy_loss = abs(buy_signals[~buy_signals['profitable']]['return'].sum()) if len(buy_signals) > 0 else 0
        sell_profit = sell_signals[sell_signals['profitable']]['return'].sum() if len(sell_signals) > 0 else 0
        sell_loss = abs(sell_signals[~sell_signals['profitable']]['return'].sum()) if len(sell_signals) > 0 else 0

        total_profit = buy_profit + sell_profit
        total_loss = buy_loss + sell_loss

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        return {
            'accuracy': profitable / total if total > 0 else 0,
            'precision': {
                'BUY': buy_precision,
                'SELL': sell_precision
            },
            'recall': {
                'BUY': buy_recall,
                'SELL': sell_recall
            },
            'avg_confidence': df['confidence'].mean(),
            'profit_factor': profit_factor,
            'total_signals': total,
            'profitable_signals': profitable,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_return': df['return'].mean(),
            'total_return': df['return'].sum()
        }

    def _empty_metrics(self) -> Dict:
        """Возвращает пустые метрики"""
        return {
            'accuracy': 0.0,
            'precision': {'BUY': 0.0, 'SELL': 0.0},
            'recall': {'BUY': 0.0, 'SELL': 0.0},
            'avg_confidence': 0.0,
            'profit_factor': 0.0,
            'total_signals': 0,
            'profitable_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'avg_return': 0.0,
            'total_return': 0.0
        }

    def check_quality(self) -> Dict:
        """
        Проверяет качество сигналов и возвращает предупреждения

        Returns:
            Словарь с результатами проверки
        """
        warnings = []
        is_critical = False

        if self.current_metrics['total_signals'] < self.alert_thresholds['min_signals']:
            return {
                'status': 'insufficient_data',
                'warnings': ['Not enough signals for analysis'],
                'is_critical': False
            }

        # Проверка точности
        if self.current_metrics['accuracy'] < self.alert_thresholds['min_accuracy']:
            warnings.append(f"Low accuracy: {self.current_metrics['accuracy']:.1%}")
            is_critical = True

        # Проверка тренда точности
        if len(self.metrics_history[f"accuracy_{min(self.window_sizes)}"]) >= 10:
            recent_accuracies = [v for _, v in list(self.metrics_history[f"accuracy_{min(self.window_sizes)}"])[-10:]]
            if len(recent_accuracies) >= 5:
                accuracy_trend = recent_accuracies[-1] - recent_accuracies[0]
                if accuracy_trend < -self.alert_thresholds['accuracy_drop']:
                    warnings.append(f"Accuracy dropping: {accuracy_trend:.1%}")
                    is_critical = True

        # Проверка уверенности
        if self.current_metrics['avg_confidence'] < 0.5:
            warnings.append(f"Low average confidence: {self.current_metrics['avg_confidence']:.1%}")

        # Проверка дисбаланса сигналов
        total = self.current_metrics['total_signals']
        buy_pct = self.current_metrics['buy_signals'] / total if total > 0 else 0
        sell_pct = self.current_metrics['sell_signals'] / total if total > 0 else 0

        if buy_pct > 0.8 or sell_pct > 0.8:
            warnings.append(f"Signal imbalance: BUY={buy_pct:.1%}, SELL={sell_pct:.1%}")

        # Проверка profit factor
        if self.current_metrics['profit_factor'] < 1.0:
            warnings.append(f"Profit factor below 1.0: {self.current_metrics['profit_factor']:.2f}")

        return {
            'status': 'critical' if is_critical else 'warning' if warnings else 'good',
            'warnings': warnings,
            'is_critical': is_critical,
            'metrics': self.current_metrics
        }

    def get_quality_score(self) -> float:
        """
        Возвращает интегральную оценку качества (0-1)

        Returns:
            Оценка качества
        """
        if self.current_metrics['total_signals'] < self.alert_thresholds['min_signals']:
            return 0.0

        score = 0.0
        weights = {
            'accuracy': 0.4,
            'confidence': 0.2,
            'profit_factor': 0.3,
            'balance': 0.1
        }

        # Точность
        accuracy_score = min(1.0, self.current_metrics['accuracy'] / 0.6)  # 60% точность = 1.0
        score += weights['accuracy'] * accuracy_score

        # Уверенность
        confidence_score = self.current_metrics['avg_confidence']
        score += weights['confidence'] * confidence_score

        # Profit factor
        pf = self.current_metrics['profit_factor']
        if pf == float('inf'):
            pf_score = 1.0
        else:
            pf_score = min(1.0, pf / 2.0)  # PF=2 = 1.0
        score += weights['profit_factor'] * pf_score

        # Баланс сигналов
        total = self.current_metrics['total_signals']
        if total > 0:
            buy_pct = self.current_metrics['buy_signals'] / total
            sell_pct = self.current_metrics['sell_signals'] / total
            balance_score = 1.0 - abs(buy_pct - 0.5) * 2  # 0.5 = идеально
            score += weights['balance'] * balance_score

        return score

    def get_trend(self, metric: str, window: int = 20) -> float:
        """
        Возвращает тренд метрики (положительный - рост, отрицательный - падение)

        Args:
            metric: название метрики
            window: размер окна

        Returns:
            Наклон тренда
        """
        history_key = f"{metric}_{min(self.window_sizes)}"
        if history_key not in self.metrics_history:
            return 0.0

        values = [v for _, v in list(self.metrics_history[history_key])[-window:]]
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def save_history(self):
        """Сохраняет историю метрик в файл"""
        history = {
            'strategy': self.strategy_name,
            'last_update': datetime.now().isoformat(),
            'signals': [
                {
                    'timestamp': s['timestamp'].isoformat(),
                    'signal': s['signal'],
                    'confidence': s['confidence'],
                    'return': s['return'],
                    'profitable': s['profitable']
                }
                for s in self.signals
            ],
            'metrics_history': {
                k: [(t.isoformat(), v) for t, v in list(vals)]
                for k, vals in self.metrics_history.items()
            }
        }

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def load_history(self):
        """Загружает историю метрик из файла"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)

            # Загружаем сигналы
            for s in history.get('signals', []):
                self.signals.append({
                    'timestamp': datetime.fromisoformat(s['timestamp']),
                    'signal': s['signal'],
                    'confidence': s['confidence'],
                    'return': s['return'],
                    'profitable': s['profitable']
                })

            # Загружаем историю метрик
            for k, vals in history.get('metrics_history', {}).items():
                self.metrics_history[k] = deque(
                    [(datetime.fromisoformat(t), v) for t, v in vals],
                    maxlen=100
                )

            # Обновляем метрики
            self._update_metrics()

        except Exception as e:
            print(f"⚠️ Ошибка загрузки истории: {e}")

    def get_report(self) -> str:
        """
        Возвращает текстовый отчет о качестве сигналов

        Returns:
            Отформатированный отчет
        """
        m = self.current_metrics

        report = []
        report.append(f"\n📊 Отчет о качестве сигналов - {self.strategy_name}")
        report.append("=" * 50)
        report.append(f"Всего сигналов: {m['total_signals']}")
        report.append(f"Прибыльных сигналов: {m['profitable_signals']} ({m['accuracy']:.1%})")
        report.append(f"Средняя уверенность: {m['avg_confidence']:.1%}")
        report.append(f"Profit Factor: {m['profit_factor']:.2f}")
        report.append(f"\nДетали по типам:")
        report.append(f"  BUY: {m['buy_signals']} сигналов, точность: {m['precision']['BUY']:.1%}")
        report.append(f"  SELL: {m['sell_signals']} сигналов, точность: {m['precision']['SELL']:.1%}")

        # Тренды
        accuracy_trend = self.get_trend('accuracy')
        if abs(accuracy_trend) > 0.001:
            trend_emoji = "📈" if accuracy_trend > 0 else "📉"
            report.append(f"\nТренд точности: {trend_emoji} {accuracy_trend:+.3f} в день")

        # Оценка качества
        score = self.get_quality_score()
        if score > 0.7:
            quality = "🟢 Отлично"
        elif score > 0.5:
            quality = "🟡 Средне"
        elif score > 0.3:
            quality = "🟠 Ниже среднего"
        else:
            quality = "🔴 Плохо"

        report.append(f"\nИнтегральная оценка: {quality} ({score:.2f})")

        return "\n".join(report)