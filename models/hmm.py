import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

def run_model(input_csv, output_path, n_states=3, sequence_length=1, return_forecast=False):
    """
    Hidden Markov Model forecast for time series data.

    Parameters:
    - input_csv: path to CSV containing Date and Price columns
    - output_path: path to save forecast plot
    - n_states: number of hidden states
    - sequence_length: how many past observations to consider (currently 1)
    - return_forecast: if True, returns forecasted values as numpy array
    """

    # -------------------------------
    # Step 1: Load and prepare data
    # -------------------------------
    df = pd.read_csv(input_csv)
    if 'S&P500' in df.columns:
        df.rename(columns={'S&P500': 'Price'}, inplace=True)

    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price']).sort_values('Date')

    series = df['Price'].values.reshape(-1, 1)

    # -------------------------------
    # Step 2: Scale data
    # -------------------------------
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series)

    # -------------------------------
    # Step 3: Fit HMM
    # -------------------------------
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    model.fit(scaled_series)

    # -------------------------------
    # Step 4: Forecast next steps
    # -------------------------------
    forecast_steps = 30
    last_obs = scaled_series[-sequence_length:].reshape(sequence_length, 1)
    forecast_scaled = []

    for _ in range(forecast_steps):
        next_state_prob = model.transmat_[model.predict(last_obs)[-1]]
        next_state = np.argmax(next_state_prob)
        # Predict next observation as mean of next state
        next_obs = model.means_[next_state]
        forecast_scaled.append(next_obs[0])
        last_obs = np.array([[next_obs[0]]])

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled)

    # -------------------------------
    # Step 5: Plot
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Price'], label="Actual", color="blue")
    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_steps+1, freq='B')[1:]
    plt.plot(forecast_dates, forecast, label="HMM Forecast", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("HMM Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # -------------------------------
    # Step 6: Print summary table
    # -------------------------------
    table_data = [[forecast_dates[i].date(), forecast[i][0]] for i in range(forecast_steps)]
    print("\nHMM FORECAST SUMMARY")
    print(tabulate(table_data, headers=["Date", "Forecast"], tablefmt="grid"))

    # -------------------------------
    # Step 7: Save CSV
    # -------------------------------
    report_path = os.path.splitext(output_path)[0] + "_forecast.csv"
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast.flatten()})
    forecast_df.to_csv(report_path, index=False)
    print(f"\nForecast saved at: {report_path}")
    print(f"Graph saved at: {output_path}")

    if return_forecast:
        return forecast.flatten()
