import pandas as pd
from typing import Dict


class MarketDataEngine:
    """
    Движок рыночных данных.

    Отвечает за:
    - загрузку исторических данных
    - хранение данных в памяти
    - обновление новыми свечами
    """

    def __init__(self, data_fetcher):

        self.fetcher = data_fetcher

        # Кэш данных
        # { "SBER": DataFrame }
        self.cache: Dict[str, pd.DataFrame] = {}

    # ==============================
    # ЗАГРУЗКА ИСТОРИИ
    # ==============================

    def load_history(self, instrument: str, days: int = 30):

        info = self.fetcher.get_instrument_info(instrument)

        if not info:
            raise Exception(f"Instrument {instrument} not found")

        figi = info["figi"]

        print(f"📥 Loading history for {instrument}")

        df = self.fetcher.fetch_candles(figi, days_back=days)

        if df.empty:
            raise Exception("No historical data")

        # убеждаемся что индекс — время
        if not isinstance(df.index, pd.DatetimeIndex):
            raise Exception("DataFrame must use DatetimeIndex")

        self.cache[instrument] = df.copy()

        print(f"✅ History loaded: {len(df)} candles")

    # ==============================
    # ОБНОВЛЕНИЕ РЫНКА
    # ==============================

    def update_market_data(self, instrument: str):

        if instrument not in self.cache:
            print(f"⚠️ No data for {instrument}")
            return

        info = self.fetcher.get_instrument_info(instrument)
        figi = info["figi"]

        df = self.cache[instrument]

        last_time = df.index[-1]

        latest = self.fetcher.fetch_recent_candles(
            figi,
            last_time=last_time
        )

        if latest.empty:
            return

        for idx, row in latest.iterrows():

            # новая свеча
            if idx > df.index[-1]:

                df.loc[idx] = row

            # обновление текущей свечи
            else:

                df.loc[idx] = row

        df.sort_index(inplace=True)

    # ==============================
    # ПОЛУЧЕНИЕ ДАННЫХ
    # ==============================

    def get_last_candle(self, instrument: str):

        if instrument not in self.cache:
            return None

        df = self.cache[instrument]

        row = df.iloc[-1]

        return {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"]
        }

    def get_dataframe(self, instrument: str):

        if instrument not in self.cache:
            return None

        return self.cache[instrument]

    def get_last_price(self, instrument: str):

        candle = self.get_last_candle(instrument)

        if candle is None:
            return None

        return candle["close"]