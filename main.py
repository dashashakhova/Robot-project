import time

from dotenv import load_dotenv
import os

from core.data_fetcher import DataFetcher
from core.broker_adapter import BrokerAdapter

from strategies.trend_follower import TrendFollowerStrategy

from execution.execution_engine import ExecutionEngine
from risk.risk_manager import RiskManager

from engine.market_data import MarketDataEngine

load_dotenv()
TOKEN = os.getenv("INVEST_TOKEN")

def main():

    instrument = "SBER"

    # DATA
    data_fetcher = DataFetcher(token=TOKEN)

    # Market Data Engine
    market_data = MarketDataEngine(data_fetcher)

    # Загружаем историю (ВАЖНО: после создания market_data)
    market_data.load_history(instrument, days=5)

    # STRATEGY
    strategy = TrendFollowerStrategy(instrument)

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
            market_data.update_market_data(instrument)

            df = market_data.get_dataframe(instrument)

            if market_data is None:
                time.sleep(5)
                continue

            price = df['close'].iloc[-1]

            # Генерируем сигнал
            signal = strategy.get_signal(df)

            if signal:

                execution_engine.execute_signal(
                    instrument=instrument,
                    signal=signal,
                    price=price
                )

            time.sleep(30)

        except Exception as e:

            print("ERROR:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()