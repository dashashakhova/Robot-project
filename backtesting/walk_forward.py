"""
Модуль для walk-forward оптимизации и валидации стратегий
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import logging

from strategies.base_strategy import BaseStrategy
from backtesting.backtester import AdvancedBacktester
from core.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    """
    Walk-forward анализ для проверки стабильности стратегии
    """

    def __init__(self, strategy: BaseStrategy, n_splits: int = 5,
                 train_size: float = 0.7, test_size: float = 0.3):
        """
        Args:
            strategy: торговая стратегия
            n_splits: количество разбиений
            train_size: доля обучающих данных
            test_size: доля тестовых данных
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size

        self.results = []
        self.oot_results = []  # Out-of-sample результаты

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Запускает walk-forward анализ

        Args:
            df: DataFrame с данными

        Returns:
            Словарь с результатами
        """
        logger.info(f"\n🔄 Запуск Walk-Forward анализа для {self.strategy.instrument}")
        logger.info(f"   Разбиений: {self.n_splits}")
        logger.info(f"   Всего данных: {len(df)} свечей")

        # Создаем временные разбиения
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            logger.info(f"\n📊 Фолд {fold + 1}/{self.n_splits}")

            train_data = df.iloc[train_idx].copy()
            test_data = df.iloc[test_idx].copy()

            logger.info(f"   Train: {train_data.index[0]} - {train_data.index[-1]} ({len(train_data)} свечей)")
            logger.info(f"   Test:  {test_data.index[0]} - {test_data.index[-1]} ({len(test_data)} свечей)")

            # Обучаем на train
            logger.info("   Обучение стратегии...")
            train_metrics = self.strategy.train(train_data, optimize=True)

            # Тестируем на test
            logger.info("   Тестирование на out-of-sample...")
            backtester = AdvancedBacktester(self.strategy, initial_capital=1000000)
            test_results = backtester.run(test_data, show_progress=False)

            # Сохраняем результаты
            fold_result = {
                'fold': fold + 1,
                'train_period': f"{train_data.index[0]} - {train_data.index[-1]}",
                'test_period': f"{test_data.index[0]} - {test_data.index[-1]}",
                'train_metrics': train_metrics,
                'test_metrics': test_results,
                'n_train': len(train_data),
                'n_test': len(test_data)
            }

            fold_results.append(fold_result)

            logger.info(f"   ✅ Тестовая доходность: {test_results.get('total_return', 0):.2%}")
            logger.info(f"   ✅ Тестовый Sharpe: {test_results.get('sharpe_ratio', 0):.2f}")

        # Агрегируем результаты
        summary = self._aggregate_results(fold_results)

        logger.info(f"\n📈 Итоги Walk-Forward анализа:")
        logger.info(f"   Средняя доходность (OOS): {summary['avg_oos_return']:.2%}")
        logger.info(f"   Стабильность доходности: {summary['return_stability']:.2%}")
        logger.info(f"   Средний Sharpe (OOS): {summary['avg_oos_sharpe']:.2f}")
        logger.info(f"   Процент прибыльных фолдов: {summary['profitable_folds']:.1%}")

        return summary

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """
        Агрегирует результаты по всем фолдам

        Args:
            fold_results: список результатов по фолдам

        Returns:
            Агрегированные результаты
        """
        if not fold_results:
            return {}

        # Собираем метрики
        oos_returns = [r['test_metrics'].get('total_return', 0) for r in fold_results]
        oos_sharpes = [r['test_metrics'].get('sharpe_ratio', 0) for r in fold_results]
        oos_drawdowns = [r['test_metrics'].get('max_drawdown', 0) for r in fold_results]
        oos_trades = [r['test_metrics'].get('total_trades', 0) for r in fold_results]
        oos_win_rates = [r['test_metrics'].get('win_rate', 0) for r in fold_results]

        # Статистика
        summary = {
            'n_folds': len(fold_results),
            'avg_oos_return': np.mean(oos_returns),
            'std_oos_return': np.std(oos_returns),
            'return_stability': 1 - (np.std(oos_returns) / abs(np.mean(oos_returns))) if abs(
                np.mean(oos_returns)) > 0 else 0,
            'max_oos_return': max(oos_returns),
            'min_oos_return': min(oos_returns),
            'profitable_folds': sum(1 for r in oos_returns if r > 0) / len(oos_returns),

            'avg_oos_sharpe': np.mean(oos_sharpes),
            'std_oos_sharpe': np.std(oos_sharpes),
            'max_oos_sharpe': max(oos_sharpes),
            'min_oos_sharpe': min(oos_sharpes),

            'avg_oos_drawdown': np.mean(oos_drawdowns),
            'max_oos_drawdown': max(oos_drawdowns),

            'avg_oos_trades': np.mean(oos_trades),
            'total_oos_trades': sum(oos_trades),
            'avg_oos_win_rate': np.mean(oos_win_rates),

            'fold_results': fold_results
        }

        # Добавляем информацию о стабильности
        if summary['avg_oos_sharpe'] > 1.0 and summary['return_stability'] > 0.5:
            summary['stability_grade'] = 'A - Высокая стабильность'
        elif summary['avg_oos_sharpe'] > 0.5 and summary['return_stability'] > 0.3:
            summary['stability_grade'] = 'B - Средняя стабильность'
        elif summary['avg_oos_sharpe'] > 0:
            summary['stability_grade'] = 'C - Низкая стабильность'
        else:
            summary['stability_grade'] = 'D - Нестабильная стратегия'

        return summary

    def plot_results(self, save_path: Optional[str] = None):
        """
        Визуализирует результаты walk-forward анализа

        Args:
            save_path: путь для сохранения
        """
        if not self.results:
            logger.warning("Нет результатов для визуализации")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Walk-Forward анализ {self.strategy.instrument}', fontsize=16)

        # Извлекаем данные
        folds = [r['fold'] for r in self.results]
        oos_returns = [r['test_metrics'].get('total_return', 0) * 100 for r in self.results]
        oos_sharpes = [r['test_metrics'].get('sharpe_ratio', 0) for r in self.results]
        oos_drawdowns = [r['test_metrics'].get('max_drawdown', 0) * 100 for r in self.results]
        oos_trades = [r['test_metrics'].get('total_trades', 0) for r in self.results]

        # 1. Доходность по фолдам
        ax1 = axes[0, 0]
        colors = ['green' if r > 0 else 'red' for r in oos_returns]
        ax1.bar(folds, oos_returns, color=colors, alpha=0.7)
        ax1.axhline(y=np.mean(oos_returns), color='blue', linestyle='--', label=f'Средняя: {np.mean(oos_returns):.1f}%')
        ax1.set_title('Out-of-sample доходность по фолдам')
        ax1.set_xlabel('Фолд')
        ax1.set_ylabel('Доходность %')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Коэффициент Шарпа
        ax2 = axes[0, 1]
        ax2.plot(folds, oos_sharpes, marker='o', color='purple', linewidth=2)
        ax2.axhline(y=np.mean(oos_sharpes), color='blue', linestyle='--', label=f'Средний: {np.mean(oos_sharpes):.2f}')
        ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Целевой уровень')
        ax2.set_title('Коэффициент Шарпа')
        ax2.set_xlabel('Фолд')
        ax2.set_ylabel('Sharpe')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Максимальная просадка
        ax3 = axes[1, 0]
        ax3.bar(folds, oos_drawdowns, color='red', alpha=0.5)
        ax3.axhline(y=np.mean(oos_drawdowns), color='blue', linestyle='--',
                    label=f'Средняя: {np.mean(oos_drawdowns):.1f}%')
        ax3.set_title('Максимальная просадка')
        ax3.set_xlabel('Фолд')
        ax3.set_ylabel('Просадка %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Количество сделок
        ax4 = axes[1, 1]
        ax4.bar(folds, oos_trades, color='orange', alpha=0.7)
        ax4.axhline(y=np.mean(oos_trades), color='blue', linestyle='--', label=f'Среднее: {np.mean(oos_trades):.0f}')
        ax4.set_title('Количество сделок')
        ax4.set_xlabel('Фолд')
        ax4.set_ylabel('Сделок')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 График сохранен в {save_path}")

        plt.show()

    def get_best_parameters(self) -> Dict:
        """
        Возвращает лучшие параметры на основе walk-forward анализа

        Returns:
            Словарь с лучшими параметрами
        """
        if not self.results:
            return {}

        # Находим фолд с лучшим Sharpe
        best_fold = max(self.results, key=lambda x: x['test_metrics'].get('sharpe_ratio', 0))

        # Получаем параметры из этого фолда
        if 'train_metrics' in best_fold and 'best_params' in best_fold['train_metrics']:
            return best_fold['train_metrics']['best_params']

        return {}

    def validate_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Комплексная валидация стратегии

        Args:
            df: полный DataFrame с данными

        Returns:
            Результаты валидации
        """
        # 1. Walk-forward анализ
        wf_results = self.run(df)

        # 2. Проверка на переобучение
        overfitting_check = self._check_overfitting()

        # 3. Проверка стабильности
        stability_check = self._check_stability()

        return {
            'walk_forward': wf_results,
            'overfitting': overfitting_check,
            'stability': stability_check,
            'passed': overfitting_check['passed'] and stability_check['passed']
        }

    def _check_overfitting(self) -> Dict:
        """
        Проверяет наличие переобучения

        Returns:
            Результаты проверки
        """
        if not self.results:
            return {'passed': False, 'reason': 'No results'}

        # Сравниваем train и test результаты
        train_sharpes = []
        test_sharpes = []

        for r in self.results:
            if 'train_metrics' in r and 'test_metrics' in r:
                train_sharpes.append(r['train_metrics'].get('sharpe_ratio', 0))
                test_sharpes.append(r['test_metrics'].get('sharpe_ratio', 0))

        if not train_sharpes or not test_sharpes:
            return {'passed': False, 'reason': 'Insufficient data'}

        avg_train_sharpe = np.mean(train_sharpes)
        avg_test_sharpe = np.mean(test_sharpes)

        # Коэффициент переобучения
        overfitting_ratio = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe > 0 else 0

        passed = overfitting_ratio > 0.5  # Тестовая эффективность > 50% от обучающей

        return {
            'passed': passed,
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'overfitting_ratio': overfitting_ratio,
            'reason': 'OK' if passed else 'Test performance too low compared to train'
        }

    def _check_stability(self) -> Dict:
        """
        Проверяет стабильность результатов

        Returns:
            Результаты проверки
        """
        if not self.results:
            return {'passed': False, 'reason': 'No results'}

        oos_returns = [r['test_metrics'].get('total_return', 0) for r in self.results]

        # Коэффициент вариации
        cv = np.std(oos_returns) / abs(np.mean(oos_returns)) if abs(np.mean(oos_returns)) > 0 else float('inf')

        # Процент прибыльных фолдов
        profitable_pct = sum(1 for r in oos_returns if r > 0) / len(oos_returns)

        passed = cv < 1.0 and profitable_pct > 0.6  # Низкая вариативность и >60% прибыльных фолдов

        return {
            'passed': passed,
            'cv': cv,
            'profitable_folds_pct': profitable_pct,
            'reason': 'OK' if passed else 'Results too volatile or too few profitable folds'
        }