"""
Модуль сбора данных из T-Invest API
Использует официальную библиотеку t-tech-investments
"""
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t_tech.invest import Client, CandleInterval
from t_tech.invest import Candle, Quotation

from core.config import DATA_DIR


class DataFetcher:
    """Класс для загрузки и сохранения исторических данных"""

    # Словарь с инструментами (актуальные FIGI из поиска)
    INSTRUMENTS = {
        # Фьючерсы (высоковолатильные, но с ограниченным сроком жизни)
        "MTLR": {"figi": "FUTMTLR09240", "name": "Мечел фьючерс", "lot": 1, "type": "futures"},
        "SMLT": {"figi": "FUTSMLT06230", "name": "Самолет фьючерс", "lot": 1, "type": "futures"},
        "AFKS": {"figi": "FUTAFKS06220", "name": "АФК Система фьючерс", "lot": 1, "type": "futures"},
        "GMKN": {"figi": "FUTGMKN06230", "name": "Норникель фьючерс", "lot": 1, "type": "futures"},

        # Акции (средняя волатильность)
        "VKCO": {"figi": "TCS00A106YF0", "name": "VK", "lot": 1, "type": "share"},
        "OZON": {"figi": "TCSC77321024", "name": "OZON", "lot": 1, "type": "share"},  # Внимание: AutoZone?
        "LKOH": {"figi": "TCS509024277", "name": "Лукойл", "lot": 1, "type": "share"},
        "YNDX": {"figi": "TCS109805522", "name": "Яндекс (Nebius)", "lot": 1, "type": "share"},
        "GAZP": {"figi": "TCS907661625", "name": "Газпром", "lot": 10, "type": "share"},
        "VTBR": {"figi": "TCS91A0JP5V6", "name": "ВТБ", "lot": 1000, "type": "share"},

        # Облигации? (нужно проверить)
        "SBER": {"figi": "BBG00XR1B5V7", "name": "Сбербанк облигация", "lot": 1, "type": "bond"},
    }

    def __init__(self, token: str):
        """
        Args:
            token: токен доступа к T-Invest API
        """
        self.token = token
        self.client = None

    def _get_client(self) -> Client:
        """Возвращает клиент API (создает при первом вызове)"""
        if self.client is None:
            self.client = Client(self.token)
        return self.client

    def fetch_candles(self, figi: str, days_back: int = 7,
                      interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN) -> pd.DataFrame:
        """
        Загружает свечи для указанного инструмента

        Args:
            figi: FIGI-идентификатор инструмента
            days_back: За сколько дней загрузить данные
            interval: Интервал свечей (по умолчанию 1 минута)

        Returns:
            DataFrame с колонками: time, open, high, low, close, volume
        """
        now = datetime.now(timezone.utc)
        from_date = now - timedelta(days=days_back)

        print(f"📥 Загрузка {figi} с {from_date.date()} по {now.date()}...")

        try:
            # Создаем новый клиент для каждого запроса
            with Client(self.token) as client:
                candles = client.get_all_candles(
                    figi=figi,
                    from_=from_date,
                    to=now,
                    interval=interval
                )

                data = []
                for candle in candles:
                    data.append(self._candle_to_dict(candle))

                if not data:
                    print(f"⚠️ Нет данных для {figi}")
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)

                print(f"✅ Загружено {len(df)} свечей")
                return df

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return pd.DataFrame()

    def fetch_recent_candles(self, figi: str, last_time: Optional[datetime] = None,
                            minutes_back: int = 60) -> pd.DataFrame:
        """
        Загружает только новые свечи с последнего известного времени

        Args:
            figi: FIGI инструмента
            last_time: время последней известной свечи
            minutes_back: если last_time нет, загрузить последние N минут

        Returns:
            DataFrame с новыми свечами
        """
        now = datetime.now(timezone.utc)

        if last_time:
            from_date = last_time
            print(f"📥 Загрузка обновлений для {figi} с {from_date.strftime('%H:%M')}...")
        else:
            from_date = now - timedelta(minutes=minutes_back)
            print(f"📥 Загрузка последних {minutes_back} минут для {figi}...")

        try:
            with Client(self.token) as client:
                candles = client.get_all_candles(
                    figi=figi,
                    from_=from_date,
                    to=now,
                    interval=CandleInterval.CANDLE_INTERVAL_1_MIN
                )

                data = []
                for candle in candles:
                    # Проверяем, не дубликат ли это
                    if last_time and candle.time <= last_time:
                        continue
                    data.append(self._candle_to_dict(candle))

                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)

                print(f"✅ Загружено {len(df)} новых свечей")
                return df

        except Exception as e:
            print(f"❌ Ошибка загрузки обновлений: {e}")
            return pd.DataFrame()

    def _candle_to_dict(self, candle: Candle) -> Dict:
        """
        Преобразует объект свечи в словарь

        Args:
            candle: объект свечи из API

        Returns:
            Словарь с данными свечи
        """
        def quotation_to_float(q: Quotation) -> float:
            """Преобразует Quotation в float"""
            return q.units + q.nano / 1e9

        return {
            'time': candle.time,
            'open': quotation_to_float(candle.open),
            'high': quotation_to_float(candle.high),
            'low': quotation_to_float(candle.low),
            'close': quotation_to_float(candle.close),
            'volume': candle.volume
        }

    def get_instrument_info(self, ticker: str) -> Optional[Dict]:
        """
        Получает информацию об инструменте по тикеру

        Args:
            ticker: тикер инструмента

        Returns:
            Словарь с информацией или None
        """
        if ticker in self.INSTRUMENTS:
            info = self.INSTRUMENTS[ticker].copy()
            info['ticker'] = ticker
            return info
        return None

    def save_to_csv(self, df: pd.DataFrame, ticker: str, clean_old: bool = False) -> Path:
        """
        Сохраняет DataFrame в CSV

        Args:
            df: DataFrame с данными
            ticker: тикер инструмента
            clean_old: удалять ли старые файлы для этого инструмента

        Returns:
            Путь к сохраненному файлу
        """
        DATA_DIR.mkdir(exist_ok=True)

        # Формируем имя файла: ТИКЕР_дата.csv
        date_str = datetime.now().strftime('%Y%m%d')
        filename = DATA_DIR / f"{ticker}_{date_str}.csv"

        # Если clean_old=True, удаляем старые файлы для этого инструмента
        if clean_old:
            for old_file in DATA_DIR.glob(f"{ticker}_*.csv"):
                if old_file != filename:
                    try:
                        old_file.unlink()
                        print(f"  🗑️ Удален старый файл: {old_file.name}")
                    except Exception as e:
                        print(f"  ⚠️ Не удалось удалить {old_file.name}: {e}")

        df.to_csv(filename)
        print(f"💾 Данные сохранены в {filename.name}")
        return filename

    def load_from_csv(self, ticker: str) -> pd.DataFrame:
        """
        Загружает последние данные из CSV для инструмента

        Args:
            ticker: тикер инструмента

        Returns:
            DataFrame с данными
        """
        # Ищем все файлы для этого инструмента
        files = list(DATA_DIR.glob(f"{ticker}_*.csv"))

        if not files:
            return pd.DataFrame()

        # Берем самый новый по дате модификации
        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        print(f"📂 Загружено {len(df)} свечей из {latest_file.name}")
        return df

    def update_csv(self, ticker: str) -> pd.DataFrame:
        """
        Обновляет CSV файл новыми данными

        Args:
            ticker: тикер инструмента

        Returns:
            Обновленный DataFrame
        """
        info = self.get_instrument_info(ticker)
        if not info:
            print(f"❌ Инструмент {ticker} не найден")
            return pd.DataFrame()

        figi = info['figi']
        DATA_DIR.mkdir(exist_ok=True)

        # Ищем существующий файл для этого инструмента
        existing_files = list(DATA_DIR.glob(f"{ticker}_*.csv"))

        if existing_files:
            # Берем самый новый файл
            latest_file = max(existing_files, key=lambda f: f.stat().st_mtime)
            df_old = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            last_time = df_old.index.max()
            print(f"📂 Последние данные в файле: {last_time.strftime('%Y-%m-%d %H:%M')}")

            # Загружаем новые данные
            df_new = self.fetch_recent_candles(figi, last_time=last_time)

            if df_new.empty:
                print(f"⚡ Нет новых данных для {ticker}")
                return df_old

            # Объединяем старые и новые данные
            df_updated = pd.concat([df_old, df_new])
            # Убираем дубликаты
            df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
            df_updated.sort_index(inplace=True)

            # Сохраняем в тот же файл
            df_updated.to_csv(latest_file)
            print(f"💾 Файл обновлен: {latest_file.name} ({len(df_updated)} свечей)")

            return df_updated
        else:
            # Если файла нет, создаем новый
            print(f"📂 Файл не найден, создаем новый...")
            df_new = self.fetch_candles(figi, days_back=7)

            if df_new.empty:
                print(f"⚠️ Не удалось загрузить данные для {ticker}")
                return pd.DataFrame()

            # Сохраняем
            self.save_to_csv(df_new, ticker, clean_old=False)
            return df_new

    def get_available_instruments(self) -> List[str]:
        """Возвращает список доступных тикеров"""
        return list(self.INSTRUMENTS.keys())

    def print_instruments_info(self):
        """Выводит информацию о доступных инструментах"""
        print("\n📊 Доступные инструменты:")
        print("=" * 90)
        print(f"{'Тикер':<8} {'Название':<25} {'FIGI':<20} {'Лот':<8} {'Тип':<12} {'Волатильность':<12}")
        print("-" * 90)

        # Группируем по типу и волатильности
        futures = ["MTLR", "SMLT", "AFKS", "GMKN"]
        shares = ["VKCO", "LKOH", "YNDX", "GAZP", "VTBR"]
        other = ["OZON", "SBER"]  # требуют проверки

        for ticker in futures:
            info = self.INSTRUMENTS[ticker]
            print(f"{ticker:<8} {info['name']:<25} {info['figi']:<20} {info['lot']:<8} {'📈 Фьючерс':<12} {'🔥 Высокая':<12}")

        for ticker in shares:
            info = self.INSTRUMENTS[ticker]
            vol = "⚡ Средняя" if ticker in ["VKCO", "YNDX"] else "💧 Низкая"
            print(f"{ticker:<8} {info['name']:<25} {info['figi']:<20} {info['lot']:<8} {'📊 Акция':<12} {vol:<12}")

        for ticker in other:
            info = self.INSTRUMENTS[ticker]
            print(f"{ticker:<8} {info['name']:<25} {info['figi']:<20} {info['lot']:<8} {'❓ Другое':<12} {'❓ Неизв.':<12}")

        print("=" * 90)
        print("\n⚠️  Внимание: Фьючерсы имеют ограниченный срок жизни (дата экспирации)")
        print("   Для торговли фьючерсами нужно учитывать дату окончания контракта")

        def get_latest_data(self, instrument):

            candles = self.get_candles(instrument)

            if not candles:
                return None

            last = candles[-1]

            return {
                "open": last.open,
                "high": last.high,
                "low": last.low,
                "close": last.close,
                "volume": last.volume
            }


def main():
    """Тестирование загрузки данных для разных инструментов"""
    from dotenv import load_dotenv
    import os

    # Загружаем токен
    load_dotenv()
    TOKEN = os.getenv('INVEST_TOKEN')

    if not TOKEN:
        print("❌ Токен не найден")
        return

    print("\n" + "=" * 90)
    print("📊 ТЕСТИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ ДЛЯ РАЗНЫХ ИНСТРУМЕНТОВ")
    print("=" * 90)

    fetcher = DataFetcher(token=TOKEN)

    # Показываем информацию об инструментах
    fetcher.print_instruments_info()

    # Тестируем загрузку для разных типов
    test_instruments = [
        ("MTLR", "Фьючерс Мечел"),
        ("VKCO", "Акция VK"),
        ("GAZP", "Акция Газпром (для сравнения)"),
    ]

    for ticker, description in test_instruments:
        print(f"\n📈 {ticker} - {description}")
        print("-" * 60)

        try:
            info = fetcher.get_instrument_info(ticker)
            if not info:
                print(f"❌ Инструмент {ticker} не найден")
                continue

            # Загружаем данные за последние 7 дней
            df = fetcher.fetch_candles(info['figi'], days_back=7)

            if not df.empty:
                print(f"✅ Загружено свечей: {len(df)}")
                print(f"📅 Период: {df.index.min()} - {df.index.max()}")

                # Рассчитываем простую волатильность
                df['returns'] = df['close'].pct_change()
                daily_vol = df['returns'].std() * 100 * (252) ** 0.5  # Годовая волатильность в %
                intraday_vol = df['returns'].std() * 100 * (252 * 24 * 60) ** 0.5  # Минутная в %

                print(f"📊 Статистика:")
                print(f"   Цена: {df['close'].min():.2f} - {df['close'].max():.2f}")
                print(f"   Средний объем: {df['volume'].mean():.0f}")
                print(f"   Волатильность (годовая): {daily_vol:.1f}%")
                print(f"   Волатильность (минутная год.): {intraday_vol:.1f}%")

                # Показываем последние 3 свечи
                print(f"\n📋 Последние 3 свечи:")
                print(df[['open', 'high', 'low', 'close', 'volume']].tail(3))
            else:
                print(f"❌ Нет данных")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 90)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 90)


if __name__ == "__main__":
    main()