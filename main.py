#!/usr/bin/env python3
"""
Главный файл запуска - УПРОЩЕННАЯ ВЕРСИЯ
Только рабочие стратегии
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import os
import argparse

import pandas as pd
from dotenv import load_dotenv

from core.data_fetcher import DataFetcher
from core.simple_backtester import SimpleBacktester
from utils.indicators import add_all_indicators

# Импортируем только простые стратегии
from strategies.level_trading import LevelTradingStrategy
from strategies.trend_follower import TrendFollowerStrategy
from strategies.simple_breakout import SimpleBreakoutStrategy

load_dotenv()
TOKEN = os.getenv('INVEST_TOKEN')

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
    parser.add_argument('--csv', type=str, help='Путь к CSV с историческими данными (open/high/low/close/volume)')

    args = parser.parse_args()
    ticker = args.instrument.upper()

    # Создаем стратегию
    strategy_class = STRATEGIES[args.strategy]
    strategy = strategy_class(ticker)

    print("=" * 70)
    print(f"🚀 {strategy.name} - {ticker}")
    print("=" * 70)

    fetcher = DataFetcher(TOKEN or "")
    df = pd.DataFrame()

    # 1) Локальный CSV всегда в приоритете
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ CSV не найден: {csv_path}")
            return
        print(f"\n📂 Загрузка данных из CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # 2) Для локального бэктеста можно работать без токена, если есть кэш
    elif args.backtest and not TOKEN:
        print("\n⚠️ Токен не найден. Пробуем загрузить локальный кэш...")
        df = fetcher.load_from_csv(ticker)
        if df.empty:
            print("❌ Локальных данных нет. Укажи --csv или добавь INVEST_TOKEN для загрузки из API.")
            return

    # 3) Во всех прочих случаях грузим из API (токен обязателен)
    else:
        if not TOKEN:
            print("❌ Токен не найден. Для онлайн-режима нужен INVEST_TOKEN.")
            print("   Для локального бэктеста можно использовать --csv.")
            return

        instrument_info = fetcher.get_instrument_info(ticker)
        if not instrument_info:
            available = ", ".join(fetcher.get_available_instruments())
            print(f"❌ Неизвестный инструмент {ticker}")
            print(f"   Доступные: {available}")
            return

        print(f"\n📥 Загрузка данных за {args.days} дней...")
        figi = instrument_info['figi']
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
