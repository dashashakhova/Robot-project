"""
Модуль управления временными задержками для интрадей торговли
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
from collections import deque

from core.config import config


class LatencyManager:
    """Управление временными задержками и синхронизацией"""

    def __init__(self):
        # Измеренные задержки (в секундах)
        self.data_fetch_latency = deque(maxlen=10)
        self.model_inference_latency = deque(maxlen=10)
        self.order_execution_latency = deque(maxlen=10)
        self.api_response_latency = deque(maxlen=10)

        # Статистика
        self.last_measure_time = None
        self.total_latency_stats = {
            'avg': 0.0,
            'max': 0.0,
            'min': float('inf'),
            'p95': 0.0
        }

        # Кэш последних цен для быстрого доступа
        self.price_cache = {}
        self.cache_timestamp = None

        # Настройки
        self.max_signal_age_seconds = 60  # Максимальный возраст сигнала
        self.max_price_change_pct = 0.001  # 0.1% максимальное изменение цены

    def measure_latency(self, operation: str, start_time: float):
        """Измеряет задержку операции"""
        latency = time.time() - start_time

        if operation == 'data_fetch':
            self.data_fetch_latency.append(latency)
        elif operation == 'inference':
            self.model_inference_latency.append(latency)
        elif operation == 'order_execution':
            self.order_execution_latency.append(latency)
        elif operation == 'api_response':
            self.api_response_latency.append(latency)

        self._update_stats()

    def _update_stats(self):
        """Обновляет статистику задержек"""
        all_latencies = (list(self.data_fetch_latency) +
                         list(self.model_inference_latency) +
                         list(self.order_execution_latency) +
                         list(self.api_response_latency))

        if all_latencies:
            self.total_latency_stats['avg'] = np.mean(all_latencies)
            self.total_latency_stats['max'] = np.max(all_latencies)
            self.total_latency_stats['min'] = np.min(all_latencies)
            self.total_latency_stats['p95'] = np.percentile(all_latencies, 95)

    def get_expected_latency(self) -> Dict[str, float]:
        """Возвращает ожидаемые задержки для разных операций"""
        return {
            'data_fetch': np.mean(self.data_fetch_latency) if self.data_fetch_latency else 2.0,
            'inference': np.mean(self.model_inference_latency) if self.model_inference_latency else 1.0,
            'execution': np.mean(self.order_execution_latency) if self.order_execution_latency else 1.0,
            'total': self.total_latency_stats['avg']
        }

    def adjust_forecast_time(self, forecast_time: datetime) -> datetime:
        """
        Корректирует время прогноза с учетом ожидаемых задержек
        """
        latencies = self.get_expected_latency()
        total_delay_seconds = latencies['total']

        # Добавляем запас в 20% на всякий случай
        adjusted_time = forecast_time + timedelta(seconds=total_delay_seconds * 1.2)
        return adjusted_time

    def is_signal_valid(self, signal_time: datetime, current_price: float,
                        signal_price: float, current_time: datetime = None) -> Tuple[bool, str]:
        """
        Проверяет, валиден ли еще сигнал с учетом временных задержек

        Returns:
            (is_valid, reason)
        """
        if current_time is None:
            current_time = datetime.now()

        # Проверяем возраст сигнала
        signal_age = (current_time - signal_time).total_seconds()

        if signal_age > self.max_signal_age_seconds:
            return False, f"Signal too old: {signal_age:.1f}s"

        # Проверяем изменение цены
        if signal_price > 0:
            price_change_pct = abs(current_price - signal_price) / signal_price
            if price_change_pct > self.max_price_change_pct:
                return False, f"Price changed too much: {price_change_pct:.3%}"

        # Проверяем, не было ли сильного движения во время задержки
        expected_latency = self.get_expected_latency()['total']
        if signal_age > expected_latency * 2:  # Если задержка больше ожидаемой в 2 раза
            return False, f"Unexpected delay: {signal_age:.1f}s > {expected_latency:.1f}s"

        return True, "Signal valid"

    def update_price_cache(self, instrument: str, price: float):
        """Обновляет кэш цен"""
        self.price_cache[instrument] = {
            'price': price,
            'timestamp': datetime.now()
        }

    def get_cached_price(self, instrument: str, max_age_seconds: float = 5.0) -> Optional[float]:
        """Получает цену из кэша, если она не устарела"""
        cached = self.price_cache.get(instrument)
        if cached:
            age = (datetime.now() - cached['timestamp']).total_seconds()
            if age <= max_age_seconds:
                return cached['price']
        return None

    def simulate_order_execution(self, signal: Dict, current_price: float) -> Dict:
        """
        Симулирует исполнение ордера с учетом задержек и проскальзывания

        Args:
            signal: сигнал на вход
            current_price: текущая рыночная цена

        Returns:
            Словарь с результатами исполнения
        """
        latencies = self.get_expected_latency()
        execution_delay = latencies['execution']

        # Симулируем движение цены за время исполнения
        # Используем нормальное распределение с волатильностью из исторических данных
        volatility = 0.0001  # Базовое значение, нужно брать из реальных данных
        price_move = np.random.normal(0, volatility * np.sqrt(execution_delay))

        if signal['action'] == 'BUY':
            execution_price = current_price * (1 + price_move + config.get('trading', 'slippage', default=0.0005))
        else:  # SELL
            execution_price = current_price * (1 - price_move - config.get('trading', 'slippage', default=0.0005))

        return {
            'execution_price': execution_price,
            'execution_time': datetime.now() + timedelta(seconds=execution_delay),
            'slippage': abs(execution_price - current_price) / current_price,
            'delay_seconds': execution_delay
        }


# Глобальный экземпляр
latency_manager = LatencyManager()