import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from tabulate import tabulate

def run_model(input_csv, output_path, model_type="lstm", sequence_length=30, epochs=50, batch_size=32, forecast_steps=20, return_forecast=False):
    # -------------------------------
    # Step 1: Load and prepare data
    # -------------------------------
    df = pd.read_csv(input_csv)
    
    # Rename S&P500 to 'Price' for consistency
    if 'S&P500' in df.columns:
        df.rename(columns={'S&P500': 'Price'}, inplace=True)
    
    # Check required columns
    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    series = df['Price'].values.reshape(-1, 1)

    # Scale series
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)

    # -------------------------------
    # Step 2: Create sequences
    # -------------------------------
    X, y = [], []
    for i in range(sequence_length, len(scaled_series)):
        X.append(scaled_series[i-sequence_length:i, 0])
        y.append(scaled_series[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 3D input for LSTM/GRU

    # -------------------------------
    # Step 3: Build model
    # -------------------------------
    model = Sequential()
    if model_type.lower() == "lstm":
        model.add(LSTM(50, activation='relu', input_shape=(sequence_length,1)))
    elif model_type.lower() == "gru":
        model.add(GRU(50, activation='relu', input_shape=(sequence_length,1)))
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # -------------------------------
    # Step 4: Train model
    # -------------------------------
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # -------------------------------
    # Step 5: Forecast
    # -------------------------------
    forecast = []
    last_seq = scaled_series[-sequence_length:].reshape(1, sequence_length, 1)

    for _ in range(forecast_steps):
        pred = model.predict(last_seq, verbose=0)[0,0]
        forecast.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)  # Fixed shape

    # Inverse scale the forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))

    # -------------------------------
    # Step 6: Plot graph
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(df['Date'], df['Price'], label="Actual", color="blue")
    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_steps+1, freq='B')[1:]
    plt.plot(forecast_dates, forecast, label="Forecast", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{model_type.upper()} Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # -------------------------------
    # Step 7: Terminal output
    # -------------------------------
    print(f"\n{model_type.upper()} Forecast Complete")
    print(f"Forecasted next {forecast_steps} business days:\n")
    for date, value in zip(forecast_dates, forecast.flatten()):
        print(f"{date.date()} : {value:.2f}")

    # -------------------------------
    # Step 8: Save forecast as CSV
    # -------------------------------
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.flatten()})
    report_path = os.path.splitext(output_path)[0] + "_forecast.csv"
    forecast_df.to_csv(report_path, index=False)
    print(f"\nForecast saved at: {report_path}")
    print(f"Graph saved at: {output_path}")

    if return_forecast:
        return forecast
