#!/usr/bin/env python3
"""
Скрипт для поиска FIGI инструментов через T-Invest API
"""
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from t_tech.invest import Client
from t_tech.invest import InstrumentIdType

# Загружаем токен
load_dotenv()
TOKEN = os.getenv('INVEST_TOKEN')

if not TOKEN:
    print("❌ Токен не найден в .env файле")
    sys.exit(1)


def find_instrument_by_ticker(ticker: str):
    """Ищет инструмент по тикеру"""
    print(f"\n🔍 Поиск инструмента {ticker}...")

    try:
        with Client(TOKEN) as client:
            # Ищем по тикеру
            instruments = client.instruments.find_instrument(query=ticker)

            if not instruments.instruments:
                print(f"   ❌ Инструмент {ticker} не найден")
                return

            print(f"\n   ✅ Найдено {len(instruments.instruments)} инструментов:")

            for i, inst in enumerate(instruments.instruments, 1):
                print(f"\n   {i}. {inst.name}")
                print(f"      Тикер: {inst.ticker}")
                print(f"      FIGI: {inst.figi}")
                print(f"      UID: {inst.uid}")
                print(f"      Тип: {inst.instrument_type}")
                print(f"      Валюта: {inst.currency if hasattr(inst, 'currency') else 'N/A'}")

                # Пытаемся получить полную информацию
                try:
                    if inst.instrument_type == "share":
                        full_info = client.instruments.share_by(
                            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                            id=inst.figi
                        )
                        if full_info.instrument:
                            print(f"      Лот: {full_info.instrument.lot}")
                except:
                    pass

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def find_all_our_instruments():
    """Ищет все нужные нам инструменты"""
    tickers = ["MTLR", "SMLT", "VKCO", "OZON", "AFKS", "GMKN", "LKOH", "YNDX", "SBER", "GAZP", "VTBR"]

    print("=" * 80)
    print("🔍 ПОИСК FIGI ДЛЯ ИНСТРУМЕНТОВ")
    print("=" * 80)

    found_instruments = {}

    for ticker in tickers:
        try:
            with Client(TOKEN) as client:
                instruments = client.instruments.find_instrument(query=ticker)

                if instruments.instruments:
                    # Берем первый подходящий (обычно это основная акция)
                    inst = instruments.instruments[0]
                    found_instruments[ticker] = {
                        'figi': inst.figi,
                        'name': inst.name,
                        'type': inst.instrument_type
                    }
                    print(f"✅ {ticker}: {inst.figi} - {inst.name}")
                else:
                    print(f"❌ {ticker}: не найден")

        except Exception as e:
            print(f"❌ {ticker}: ошибка - {e}")

    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ПОИСКА:")
    print("=" * 80)

    for ticker, info in found_instruments.items():
        print(f"{ticker}: {info['figi']}  # {info['name']}")

    return found_instruments


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Поиск FIGI инструментов')
    parser.add_argument('--ticker', type=str, help='Тикер для поиска')
    parser.add_argument('--all', action='store_true', help='Найти все инструменты')

    args = parser.parse_args()

    if args.ticker:
        find_instrument_by_ticker(args.ticker.upper())
    elif args.all:
        find_all_our_instruments()
    else:
        # Интерактивный режим
        print("1. Найти конкретный тикер")
        print("2. Найти все наши инструменты")
        print("3. Проверить FIGI из списка")

        choice = input("Выберите действие (1/2/3): ").strip()

        if choice == '1':
            ticker = input("Введите тикер: ").strip().upper()
            find_instrument_by_ticker(ticker)
        elif choice == '2':
            find_all_our_instruments()
        elif choice == '3':
            # Проверяем FIGI из нашего списка
            test_figis = {
                "MTLR": "BBG00475WR89",
                "SMLT": "BBG00P7QDN04",
                "VKCO": "BBG00QPYJ5H0",
                "OZON": "BBG00R7G3H23",
                "AFKS": "BBG004S68B05",
                "GMKN": "BBG0047315Y0",
                "LKOH": "BBG004731032",
                "YNDX": "BBG006L8G4H1",
                "SBER": "BBG004730N88",
                "GAZP": "BBG004730RP0",
                "VTBR": "BBG004730ZJ9",
            }

            print("\n🔍 Проверка FIGI из списка:")
            for ticker, figi in test_figis.items():
                try:
                    with Client(TOKEN) as client:
                        instruments = client.instruments.find_instrument(query=figi)
                        if instruments.instruments:
                            print(f"✅ {ticker}: {figi} - работает")
                        else:
                            print(f"❌ {ticker}: {figi} - не работает")
                except:
                    print(f"❌ {ticker}: {figi} - ошибка")


if __name__ == "__main__":
    main()