#!/usr/bin/env python3
"""
Главный файл запуска - УПРОЩЕННАЯ ВЕРСИЯ
Только рабочие стратегии
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import argparse
from dotenv import load_dotenv
import os
from datetime import datetime

from core.config import config
from core.data_fetcher import DataFetcher
from core.simple_backtester import SimpleBacktester
from utils.indicators import add_all_indicators

# Импортируем только простые стратегии
from strategies.level_trading import LevelTradingStrategy
from strategies.trend_follower import TrendFollowerStrategy
from strategies.simple_breakout import SimpleBreakoutStrategy

load_dotenv()
TOKEN = os.getenv('INVEST_TOKEN')

if not TOKEN:
    print("❌ Токен не найден")
    sys.exit(1)

STRATEGIES = {
    'level': LevelTradingStrategy,
    'trend': TrendFollowerStrategy,
    'breakout': SimpleBreakoutStrategy,
}


def main():
    parser = argparse.ArgumentParser(description='Простой торговый робот')
    parser.add_argument('--instrument', type=str, default='VKCO', help='Тикер')
    parser.add_argument('--strategy', type=str, choices=STRATEGIES.keys(), default='level', help='Стратегия')
    parser.add_argument('--days', type=int, default=30, help='Дней истории')
    parser.add_argument('--backtest', action='store_true', help='Запустить бэктест')
    parser.add_argument('--plot', action='store_true', help='Показать график')

    args = parser.parse_args()

    # Создаем стратегию
    strategy_class = STRATEGIES[args.strategy]
    strategy = strategy_class(args.instrument)

    print("=" * 70)
    print(f"🚀 {strategy.name} - {args.instrument}")
    print("=" * 70)

    # Загружаем данные
    fetcher = DataFetcher(TOKEN)
    print(f"\n📥 Загрузка данных за {args.days} дней...")

    # Получаем FIGI
    if args.instrument == 'VKCO':
        figi = "TCS00A106YF0"
    elif args.instrument == 'YNDX':
        figi = "TCS109805522"
    else:
        print(f"❌ Неизвестный инструмент {args.instrument}")
        return

    df = fetcher.fetch_candles(figi, days_back=args.days)

    if df.empty:
        print("❌ Нет данных")
        return

    print(f"✅ Загружено {len(df)} свечей")

    # Добавляем индикаторы
    df = add_all_indicators(df)

    if args.backtest:
        # Запускаем бэктест
        backtester = SimpleBacktester(strategy)
        results = backtester.run(df)

        if args.plot:
            backtester.plot_equity()

        print(f"\n📊 Итоги:")
        print(f"   Сделок: {results['total_trades']}")
        print(f"   Винрейт: {results['win_rate']:.1%}")
        print(f"   Доходность: {results['total_return']:.2%}")

    else:
        # Режим реального времени (упрощенно)
        print("\n🔮 Текущий сигнал:")
        signal = strategy.get_signal(df)

        if signal['signal'] != 'HOLD':
            print(f"   Действие: {signal['signal']}")
            print(f"   Уверенность: {signal['confidence']:.1%}")
            print(f"   Причина: {', '.join(signal['reason'])}")
            print(f"   Вход: {signal['entry_price']:.2f}")
            print(f"   Стоп: {signal['stop_loss']:.2f}")
            print(f"   Тейк: {signal['take_profit']:.2f}")
        else:
            print("   ⚪ НЕТ СИГНАЛА")


if __name__ == "__main__":
    main()