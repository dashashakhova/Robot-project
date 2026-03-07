#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Главный файл запуска торгового робота
T-Invest API Intraday Trading Robot
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from core.config import config, get_sandbox_account_id
from core.data_fetcher import DataFetcher
from core.latency_manager import latency_manager
from core.market_context import MarketContextAnalyzer

from strategies.ensemble_strategy import EnsembleStrategy
from strategies.ml_strategy import MLStrategy

from risk.position_sizer import PositionSizeManager
from risk.stop_loss_manager import StopLossManager
from risk.portfolio_risk import PortfolioRiskManager

from backtesting.backtester import AdvancedBacktester
from backtesting.walk_forward import WalkForwardAnalyzer

from monitoring.signal_quality import SignalQualityAnalyzer
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alerts import AlertManager

from utils.indicators import add_all_indicators
from utils.logger import setup_logging

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logger = setup_logging('trading_robot')


class TradingRobot:
    """
    Главный класс торгового робота
    """

    def __init__(self, instrument: str, figi: str, mode: str = 'sandbox'):
        """
        Args:
            instrument: тикер инструмента
            figi: FIGI инструмента
            mode: режим работы ('sandbox' или 'real')
        """
        self.instrument = instrument
        self.figi = figi
        self.mode = mode

        # Токен
        self.token = os.getenv('INVEST_TOKEN')
        if not self.token:
            raise ValueError("❌ Токен не найден в .env файле")

        # Инициализация компонентов
        logger.info(f"🚀 Инициализация торгового робота для {instrument}")
        logger.info(f"   Режим: {mode}")

        # Data Fetcher
        self.fetcher = DataFetcher(self.token)

        # Стратегия
        self.strategy = EnsembleStrategy(instrument)

        # Риск-менеджмент
        initial_capital = config.get('trading', 'initial_capital', default=1000000)
        self.position_manager = PositionSizeManager(initial_capital)
        self.stop_manager = StopLossManager(instrument)
        self.portfolio_manager = PortfolioRiskManager(initial_capital)

        # Мониторинг
        self.signal_analyzer = SignalQualityAnalyzer(instrument)
        self.performance_tracker = PerformanceTracker(initial_capital, instrument)
        self.alert_manager = AlertManager(instrument)

        # Состояние
        self.is_running = False
        self.last_update = None
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)

        # Данные
        self.data = None
        self.indicators = None

    def initialize(self, days_back: int = 60):
        """
        Инициализация робота: загрузка данных и обучение моделей

        Args:
            days_back: количество дней истории для загрузки
        """
        logger.info("🔄 Инициализация...")

        # Загружаем данные
        logger.info(f"   Загрузка данных за {days_back} дней...")
        self.data = self.fetcher.fetch_candles(self.figi, days_back=days_back)

        if self.data.empty:
            logger.error("❌ Не удалось загрузить данные")
            return False

        logger.info(f"   Загружено {len(self.data)} свечей")

        # Добавляем индикаторы
        logger.info("   Расчет индикаторов...")
        self.indicators = add_all_indicators(self.data)

        # Пытаемся загрузить сохраненные модели
        if self.strategy.load_all_models():
            logger.info("   ✅ Модели загружены из файлов")
        else:
            # Обучаем модели
            logger.info("   Обучение моделей...")
            self.strategy.train(self.indicators, optimize=True)
            self.strategy.save_all_models()
            logger.info("   ✅ Модели обучены и сохранены")

        # Проверяем качество сигналов на истории
        logger.info("   Валидация стратегии...")
        backtester = AdvancedBacktester(self.strategy, initial_capital=1000000)
        results = backtester.run(self.indicators[-1000:], show_progress=False)

        logger.info(f"   Валидация:")
        logger.info(f"     • Точность: {results.get('win_rate', 0):.1%}")
        logger.info(f"     • Доходность: {results.get('total_return', 0):+.2%}")
        logger.info(f"     • Сделок: {results.get('total_trades', 0)}")

        return True

    def update_data(self):
        """
        Обновляет данные с биржи
        """
        logger.debug("🔄 Обновление данных...")

        start_time = time.time()

        # Загружаем новые данные
        new_data = self.fetcher.update_csv(self.instrument, self.figi)

        if not new_data.empty:
            self.data = new_data
            self.indicators = add_all_indicators(self.data)
            logger.debug(f"   Данные обновлены: {len(new_data)} свечей")

        # Измеряем задержку
        latency_manager.measure_latency('data_fetch', start_time)

        self.last_update = datetime.now()

    def check_signals(self) -> Dict:
        """
        Проверяет текущие сигналы

        Returns:
            Словарь с сигналами
        """
        if self.indicators is None or len(self.indicators) < 100:
            logger.warning("⚠️ Недостаточно данных для анализа")
            return {}

        # Получаем последние данные
        last_data = self.indicators.iloc[-100:].copy()

        # Получаем сигнал от стратегии
        start_time = time.time()
        signal = self.strategy.get_signal(last_data)
        latency_manager.measure_latency('inference', start_time)

        # Проверяем валидность сигнала с учетом задержек
        if signal['action'] != 'HOLD':
            is_valid, reason = latency_manager.is_signal_valid(
                datetime.now(),
                signal.get('current_price', 0),
                signal.get('current_price', 0)
            )

            if not is_valid:
                logger.info(f"   ⚠️ Сигнал отклонен: {reason}")
                return {}

        return signal

    def execute_signal(self, signal: Dict):
        """
        Исполняет торговый сигнал

        Args:
            signal: сигнал
        """
        logger.info(f"📊 Сигнал: {signal['action']} (уверенность: {signal['confidence']:.1%})")

        # Отправляем уведомление
        self.alert_manager.send_signal_alert(signal)

        # В песочнице только логируем
        if self.mode == 'sandbox':
            logger.info(f"   🏖️ Песочница: сигнал получен, сделка не исполняется")
            return

        # TODO: Реальная торговля
        logger.info("   TODO: Реальное исполнение сделок")

        # Здесь будет код для реальной торговли через T-Invest API

    def check_positions(self):
        """
        Проверяет открытые позиции
        """
        if not self.portfolio_manager.positions:
            return

        # Получаем текущие цены
        current_prices = {}
        for instrument in self.portfolio_manager.positions.keys():
            # В реальном проекте здесь запрос текущей цены
            if instrument == self.instrument and self.indicators is not None:
                current_prices[instrument] = self.indicators['close'].iloc[-1]

        # Обновляем позиции
        self.portfolio_manager.update_positions(current_prices)

        # Проверяем стопы
        for instrument, position in self.portfolio_manager.positions.items():
            if instrument in current_prices:
                stop_check = self.stop_manager.check_stop_trigger(
                    f"pos_{instrument}",
                    current_prices[instrument]
                )

                if stop_check['triggered']:
                    # Закрываем позицию
                    trade = self.portfolio_manager.close_position(
                        instrument,
                        stop_check['price'],
                        stop_check['reason']
                    )

                    if trade:
                        # Обновляем трекер производительности
                        self.performance_tracker.add_trade(trade)

                        # Отправляем уведомление
                        self.alert_manager.send_trade_alert(trade, is_open=False)

                        # Обновляем анализатор качества сигналов
                        self.signal_analyzer.add_signal_result(
                            trade['entry_time'],
                            trade['direction'],
                            1.0,  # confidence (упрощенно)
                            trade['pnl_pct']
                        )

    def daily_reset(self):
        """
        Ежедневный сброс статистики и проверки
        """
        now = datetime.now()

        # Проверяем, нужно ли делать сброс (в полночь)
        if now.date() > self.daily_reset_time.date():
            logger.info("📅 Ежедневный сброс статистики")

            # Получаем дневную статистику
            daily_stats = {
                'daily_pnl': self.performance_tracker.today_pnl,
                'daily_trades': self.performance_tracker.today_trades,
                'daily_win_rate': self.signal_analyzer.current_metrics.get('accuracy', 0),
                'current_capital': self.performance_tracker.current_capital,
                'total_return': self.performance_tracker.metrics['total_return'],
                'win_rate': self.performance_tracker.metrics['win_rate'],
                'profit_factor': self.performance_tracker.metrics['profit_factor'],
                'max_drawdown': self.performance_tracker.metrics['max_drawdown'],
                'open_positions': len(self.portfolio_manager.positions),
                'signals_today': self.signal_analyzer.current_metrics.get('total_signals', 0)
            }

            # Отправляем отчет
            self.alert_manager.send_daily_report(daily_stats)

            # Сбрасываем дневные счетчики
            self.portfolio_manager.reset_daily_stats()
            self.daily_reset_time = now

    def run(self, interval_minutes: int = 5, max_cycles: Optional[int] = None):
        """
        Запускает основной цикл робота

        Args:
            interval_minutes: интервал проверки в минутах
            max_cycles: максимальное количество циклов (None = бесконечно)
        """
        logger.info(f"▶️ Запуск основного цикла (интервал: {interval_minutes} мин)")

        self.is_running = True
        cycle = 0

        # Отправляем тестовое сообщение
        self.alert_manager.send_test_message()

        try:
            while self.is_running:
                cycle += 1

                if max_cycles and cycle > max_cycles:
                    logger.info(f"✅ Достигнуто максимальное количество циклов ({max_cycles})")
                    break

                logger.info(f"\n🔄 Цикл {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # 1. Обновляем данные
                self.update_data()

                # 2. Проверяем открытые позиции
                self.check_positions()

                # 3. Проверяем сигналы
                signal = self.check_signals()

                # 4. Исполняем сигнал если есть
                if signal and signal['action'] != 'HOLD':
                    self.execute_signal(signal)

                # 5. Проверяем качество сигналов
                quality_check = self.signal_analyzer.check_quality()
                if quality_check['is_critical']:
                    self.alert_manager.send_critical_alert(
                        'signal_quality',
                        'Критическое падение качества сигналов',
                        quality_check
                    )
                elif quality_check['warnings']:
                    self.alert_manager.send_warning(
                        'signal_quality',
                        'Предупреждение о качестве сигналов',
                        quality_check
                    )

                # 6. Проверяем здоровье системы
                health_check = self.performance_tracker.check_health()
                if health_check['is_critical']:
                    self.alert_manager.send_critical_alert(
                        'system_health',
                        'Критическое состояние системы',
                        health_check
                    )
                    # При критическом состоянии останавливаемся
                    if health_check['is_critical']:
                        logger.error("🛑 Критическое состояние, остановка...")
                        break
                elif health_check['warnings']:
                    self.alert_manager.send_warning(
                        'system_health',
                        'Предупреждение о состоянии системы',
                        health_check
                    )

                # 7. Ежедневный сброс
                self.daily_reset()

                # 8. Обновляем метрики производительности
                self.performance_tracker.update_equity(
                    self.portfolio_manager.current_capital
                )

                # 9. Выводим статус
                self.print_status()

                # 10. Ждем до следующего цикла
                if max_cycles is None or cycle < max_cycles:
                    logger.info(f"⏳ Ожидание {interval_minutes} минут...")
                    time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("🛑 Остановлено пользователем")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
            self.alert_manager.send_critical_alert(
                'system_error',
                f'Критическая ошибка: {str(e)}'
            )
        finally:
            self.stop()

    def stop(self):
        """
        Останавливает робота
        """
        logger.info("🛑 Остановка робота...")
        self.is_running = False

        # Закрываем все позиции (в реальном режиме)
        if self.mode == 'real' and self.portfolio_manager.positions:
            logger.info("   Закрытие всех позиций...")
            # TODO: Закрытие позиций через API

        # Сохраняем историю
        self.signal_analyzer.save_history()
        self.performance_tracker.save_history()
        self.alert_manager.save_history()

        logger.info("✅ Робот остановлен")

    def print_status(self):
        """
        Выводит текущий статус
        """
        status = []
        status.append(f"\n📊 СТАТУС НА {datetime.now().strftime('%H:%M:%S')}")
        status.append("=" * 50)

        # Портфель
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        status.append(f"💰 Капитал: {portfolio_status['current_capital']:,.0f}")
        status.append(f"📈 Доходность: {portfolio_status['total_return']:+.2%}")
        status.append(f"📉 Просадка: {portfolio_status['current_drawdown']:.2%}")
        status.append(f"📊 Открытых позиций: {portfolio_status['open_positions']}")

        # Качество сигналов
        quality_score = self.signal_analyzer.get_quality_score()
        quality_emoji = '🟢' if quality_score > 0.7 else '🟡' if quality_score > 0.4 else '🔴'
        status.append(f"\n{quality_emoji} Качество сигналов: {quality_score:.2f}")

        # Последние метрики
        status.append(f"\n🎯 Метрики:")
        status.append(f"  Винрейт: {self.performance_tracker.metrics['win_rate']:.1%}")
        status.append(f"  Profit Factor: {self.performance_tracker.metrics['profit_factor']:.2f}")
        status.append(f"  Sharpe: {self.performance_tracker.metrics['sharpe_ratio']:.2f}")

        # Задержки
        latencies = latency_manager.get_expected_latency()
        status.append(f"\n⏱️ Задержки:")
        status.append(f"  Данные: {latencies['data_fetch']:.2f}с")
        status.append(f"  Инференс: {latencies['inference']:.2f}с")

        logger.info("\n".join(status))


def setup_argparse():
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(description='T-Invest Intraday Trading Robot')

    parser.add_argument('--instrument', type=str, default='SBER',
                        help='Тикер инструмента (по умолчанию: SBER)')

    parser.add_argument('--figi', type=str,
                        help='FIGI инструмента (если не указан, будет найден автоматически)')

    parser.add_argument('--mode', type=str, choices=['sandbox', 'real'], default='sandbox',
                        help='Режим работы: sandbox или real (по умолчанию: sandbox)')

    parser.add_argument('--interval', type=int, default=5,
                        help='Интервал проверки в минутах (по умолчанию: 5)')

    parser.add_argument('--cycles', type=int, default=None,
                        help='Количество циклов для выполнения (по умолчанию: бесконечно)')

    parser.add_argument('--backtest', action='store_true',
                        help='Запустить бэктестинг вместо реальной торговли')

    parser.add_argument('--backtest-days', type=int, default=30,
                        help='Количество дней для бэктестинга (по умолчанию: 30)')

    parser.add_argument('--walk-forward', action='store_true',
                        help='Выполнить walk-forward анализ')

    parser.add_argument('--train', action='store_true',
                        help='Переобучить модели')

    parser.add_argument('--test-alerts', action='store_true',
                        help='Отправить тестовое уведомление')

    return parser


def main():
    """Главная функция"""
    parser = setup_argparse()
    args = parser.parse_args()

    # Проверяем токен
    token = os.getenv('INVEST_TOKEN')
    if not token:
        print("❌ Токен не найден. Добавьте INVEST_TOKEN в .env файл")
        return

    # Получаем ID счета в песочнице
    if args.mode == 'sandbox':
        account_id = get_sandbox_account_id()
        if account_id:
            logger.info(f"🏦 Используем счет в песочнице: {account_id}")
        else:
            logger.warning("⚠️ Счет в песочнице не найден. Запустите sandbox_setup.py")

    # Если FIGI не указан, пытаемся найти
    figi = args.figi
    if not figi:
        from utils.figi_lookup import FIGILookup
        lookup = FIGILookup(token)
        figi = lookup.find_with_retry(args.instrument)

        if not figi:
            print(f"❌ Не удалось найти FIGI для {args.instrument}")
            return

        print(f"✅ Найден FIGI: {figi}")

    # Создаем робота
    robot = TradingRobot(args.instrument, figi, args.mode)

    # Тестовое уведомление
    if args.test_alerts:
        robot.alert_manager.send_test_message()
        return

    # Обучение
    if args.train:
        robot.initialize(days_back=60)
        return

    # Walk-forward анализ
    if args.walk_forward:
        print("\n🔄 Walk-Forward анализ")
        robot.initialize(days_back=90)
        wf = WalkForwardAnalyzer(robot.strategy)
        results = wf.run(robot.indicators)
        print(f"\nРезультаты: {results}")
        return

    # Бэктестинг
    if args.backtest:
        print("\n📊 Бэктестинг")
        robot.initialize(days_back=args.backtest_days + 30)

        backtester = AdvancedBacktester(robot.strategy)
        results = backtester.run(
            robot.indicators[-args.backtest_days * 24:],  # примерно N дней по 24 часа
            show_progress=True
        )

        # Сохраняем результаты
        backtester.save_results()

        # Строим графики
        backtester.plot_results(save_path=f"backtest_{args.instrument}.png")

        return

    # Реальная торговля
    print(f"\n🚀 Запуск торгового робота в режиме {args.mode}")

    # Инициализация
    if not robot.initialize(days_back=60):
        print("❌ Ошибка инициализации")
        return

    # Запуск основного цикла
    robot.run(interval_minutes=args.interval, max_cycles=args.cycles)


if __name__ == "__main__":
    main()