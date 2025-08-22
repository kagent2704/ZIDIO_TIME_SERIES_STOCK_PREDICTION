import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os


def run_model(input_csv, output_path, model_type="arima", forecast_steps=20, return_forecast=False):
    # -------------------------------
    # Step 1: Load and prepare data
    # -------------------------------
    df = pd.read_csv(input_csv)

    # Auto-detect date column
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "timestamp"]:
            date_col = col
            break

    if date_col is None:
        raise ValueError("CSV must contain a 'Date' or 'Timestamp' column")

    # Auto-detect price column (supports S&P500, Close, Adj Close, Price, etc.)
    price_col = None
    for col in df.columns:
        if col.lower() in ["price", "s&p500", "close", "adj close", "value"]:
            price_col = col
            break

    if price_col is None:
        raise ValueError("CSV must contain a price column (e.g., 'Price', 'S&P500', 'Close')")

    # Rename for consistency
    df = df.rename(columns={date_col: "date", price_col: "price"})

    # Convert date and clean data
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date', 'price']).sort_values('date')

    # Set index and frequency
    df = df.set_index('date')
    df = df.asfreq("B")  # Business day frequency
    series = df['price']

    # -------------------------------
    # Step 2: Train Model
    # -------------------------------
    if model_type == "arima":
        model = ARIMA(series, order=(5, 1, 0))
    elif model_type == "sarima":
        model = SARIMAX(series, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    else:
        raise ValueError("Invalid model_type. Choose 'arima' or 'sarima'.")

    fitted_model = model.fit()

    # -------------------------------
    # Step 3: Forecast
    # -------------------------------
    forecast_steps = 30
    forecast = fitted_model.forecast(steps=forecast_steps)

    # Generate future dates to match forecast
    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=forecast_steps+1, freq="B")[1:]
    forecast.index = forecast_index

    # -------------------------------
    # Step 4: Save results
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series, label="Historical", color="blue")
    plt.plot(forecast.index, forecast, label="Forecast", color="red", linewidth=2)
    plt.legend()
    plt.title(f"{model_type.upper()} Forecast")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Print summary in terminal
    print("\n" + "="*60)
    print(f" {model_type.upper()} MODEL SUMMARY ")
    print("="*60)
    print(fitted_model.summary())
    print("\nForecast (next 30 business days):")
    print(forecast)
    print("="*60)

    if return_forecast:
        return forecast
