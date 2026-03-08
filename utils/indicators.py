"""
Модуль с техническими индикаторами для анализа
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Простое скользящее среднее (SMA)"""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Экспоненциальное скользящее среднее (EMA)"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Индекс относительной силы (RSI)

    RSI = 100 - (100 / (1 + RS))
    RS = средний рост / среднее падение за период
    """
    delta = data.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Избегаем деления на ноль
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)

    Returns:
        MACD линия, сигнальная линия, гистограмма
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Полосы Боллинджера

    Returns:
        Верхняя полоса, средняя линия (SMA), нижняя полоса
    """
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Скользящее среднее объема"""
    return volume.rolling(window=period).mean()


def calculate_volume_surge(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Отношение текущего объема к среднему объему за период

    Значение > 1 означает, что объем выше среднего
    Значение < 1 означает, что объем ниже среднего
    """
    volume_avg = calculate_volume_sma(volume, period)
    return volume / volume_avg


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range - волатильность

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_price_range(df: pd.DataFrame) -> pd.Series:
    """
    Волатильность свечи в процентах от цены закрытия
    """
    return (df['high'] - df['low']) / df['close'] * 100


def calculate_momentum(close: pd.Series, period: int = 5) -> pd.Series:
    """
    Моментум - изменение цены за N периодов
    """
    return close - close.shift(period)


def calculate_acceleration(momentum: pd.Series, period: int = 5) -> pd.Series:
    """
    Ускорение - изменение моментума
    """
    return momentum - momentum.shift(period)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV)
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0

    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = pd.Series(0, index=df.index)
    negative_flow = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        else:
            negative_flow.iloc[i] = money_flow.iloc[i]

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    money_ratio = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Стохастический осциллятор

    Returns:
        %K, %D
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()

    return k, d


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()

    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

    return williams_r


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - sma_tp) / (0.015 * mad)

    return cci


def add_all_indicators(df: pd.DataFrame,
                       sma_periods: List[int] = [5, 10, 20, 50],
                       ema_periods: List[int] = [12, 26],
                       rsi_period: int = 14,
                       macd_fast: int = 12,
                       macd_slow: int = 26,
                       macd_signal: int = 9,
                       bb_period: int = 20,
                       bb_std: float = 2.0,
                       volume_period: int = 20,
                       atr_period: int = 14,
                       momentum_period: int = 5) -> pd.DataFrame:
    """
    Добавляет все технические индикаторы в DataFrame

    Args:
        df: DataFrame с колонками open, high, low, close, volume
        sma_periods: список периодов для SMA
        ema_periods: список периодов для EMA
        rsi_period: период для RSI
        macd_fast: быстрый период MACD
        macd_slow: медленный период MACD
        macd_signal: период сигнальной линии MACD
        bb_period: период для полос Боллинджера
        bb_std: количество стандартных отклонений для полос Боллинджера
        volume_period: период для скользящего среднего объема
        atr_period: период для ATR
        momentum_period: период для моментума

    Returns:
        DataFrame с добавленными индикаторами
    """
    result = df.copy()

    # SMA
    for period in sma_periods:
        result[f'sma_{period}'] = calculate_sma(result['close'], period)

    # EMA
    for period in ema_periods:
        result[f'ema_{period}'] = calculate_ema(result['close'], period)

    # RSI
    result['rsi'] = calculate_rsi(result['close'], rsi_period)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(
        result['close'], macd_fast, macd_slow, macd_signal
    )
    result['macd_line'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        result['close'], bb_period, bb_std
    )
    result['bb_upper'] = upper
    result['bb_middle'] = middle
    result['bb_lower'] = lower
    result['bb_width'] = (upper - lower) / middle  # Ширина полос (нормализованная)
    result['bb_position'] = (result['close'] - lower) / (upper - lower)  # Позиция цены в полосах

    # Объем
    result['volume_sma'] = calculate_volume_sma(result['volume'], volume_period)
    result['volume_ratio'] = result['volume'] / result['volume_sma']  # Отношение объема к среднему
    result['volume_surge'] = calculate_volume_surge(result['volume'], volume_period)  # Явный индикатор всплеска

    # ATR (волатильность)
    result['atr'] = calculate_atr(result, atr_period)
    result['atr_percent'] = result['atr'] / result['close'] * 100  # ATR в процентах от цены

    # Дополнительные индикаторы волатильности
    result['price_range'] = calculate_price_range(result)
    result['momentum'] = calculate_momentum(result['close'], momentum_period)
    result['acceleration'] = calculate_acceleration(result['momentum'], momentum_period)

    # OBV (On-Balance Volume)
    result['obv'] = calculate_obv(result)

    # MFI (Money Flow Index)
    result['mfi'] = calculate_mfi(result, period=14)

    # Стохастик
    k, d = calculate_stochastic(result)
    result['stoch_k'] = k
    result['stoch_d'] = d

    # Williams %R
    result['williams_r'] = calculate_williams_r(result)

    # CCI
    result['cci'] = calculate_cci(result)

    # Дополнительные признаки
    # Логарифмическая доходность
    result['log_return'] = np.log(result['close'] / result['close'].shift(1))

    # Отношение цены к SMA
    for period in sma_periods:
        result[f'close_to_sma_{period}'] = result['close'] / result[f'sma_{period}']

    return result


# Для тестирования
if __name__ == "__main__":
    # Создаем тестовые данные
    dates = pd.date_range('2026-01-01', '2026-02-01', freq='1min')
    np.random.seed(42)

    test_df = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 10 + 100,
        'high': np.random.randn(len(dates)) * 10 + 102,
        'low': np.random.randn(len(dates)) * 10 + 98,
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print("📊 Тестовые данные созданы")
    print(f"   Форма: {test_df.shape}")
    print(f"   Диапазон: {test_df.index.min()} - {test_df.index.max()}")

    # Добавляем индикаторы
    df_with_indicators = add_all_indicators(test_df)

    print("\n✅ Индикаторы добавлены")
    print(f"   Новые колонки: {list(df_with_indicators.columns)}")
    print(f"\nПервые строки с индикаторами:")
    print(df_with_indicators[['close', 'sma_20', 'rsi', 'volume_surge']].head())