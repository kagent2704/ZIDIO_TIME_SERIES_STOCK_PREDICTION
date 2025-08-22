import pandas as pd
import ta  # technical analysis library

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators: RSI, MACD, Bollinger Bands
    df should contain 'Close' price column (and ideally 'High', 'Low')
    """
    df = df.copy()

    # === RSI ===
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # === MACD ===
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    # === Bollinger Bands ===
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["Bollinger_High"] = bollinger.bollinger_hband()
    df["Bollinger_Low"] = bollinger.bollinger_lband()
    df["Bollinger_Mavg"] = bollinger.bollinger_mavg()

    return df
