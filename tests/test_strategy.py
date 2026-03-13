from dotenv import load_dotenv
import os

from core.data_fetcher import DataFetcher
from engine.market_data import MarketDataEngine
from strategies.trend_follower import TrendFollowerStrategy

load_dotenv()
TOKEN = os.getenv("INVEST_TOKEN")


def test_strategy():

    fetcher = DataFetcher(TOKEN)
    market = MarketDataEngine(fetcher)

    instrument = "SBER"

    market.load_history(instrument, days=5)

    df = market.get_dataframe(instrument)

    strategy = TrendFollowerStrategy(instrument)

    signal = strategy.get_signal(df)

    print("Strategy output:")
    print(signal)


if __name__ == "__main__":
    test_strategy()
