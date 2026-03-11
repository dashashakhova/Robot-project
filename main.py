import time

from dotenv import load_dotenv
import os

from core.data_fetcher import DataFetcher
from core.broker_adapter import BrokerAdapter

from strategies.trend_follower import TrendFollowerStrategy

from execution.execution_engine import ExecutionEngine
from risk.risk_manager import RiskManager

load_dotenv()
TOKEN = os.getenv("t.WoukfFLjVlRd5PeZ0gne4_pGkjGu7VGXG1RfUAYHQ3JUW2F3_3WV16r2xHjjsxPT-0F0NgQyZIXoMKp2F_u0HQ")

fetcher = DataFetcher(token=TOKEN)


def main():

    instrument = "SBER"

    # DATA
    data_fetcher = DataFetcher()

    # STRATEGY
    strategy = TrendFollowerStrategy()

    # RISK
    risk_manager = RiskManager()

    # BROKER
    broker = BrokerAdapter(None, None)

    # EXECUTION
    execution_engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager
    )

    print("TRADING BOT STARTED")

    while True:

        try:

            # Получаем данные
            market_data = data_fetcher.get_latest_data(instrument)

            if market_data is None:
                time.sleep(5)
                continue

            price = market_data["close"]

            # Генерируем сигнал
            signal = strategy.generate_signal(market_data)

            if signal:

                execution_engine.execute_signal(
                    instrument=instrument,
                    signal=signal,
                    price=price
                )

            time.sleep(10)

        except Exception as e:

            print("ERROR:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()