import os
import numpy as np
import pandas as pd
from models import (
    arima_sarima, lstm_gru, ensemble, hmm, anomaly,
    sentiment, rl_agent, gans, shap_lime, transformer_forecaster
)


input_csv = "data/sp500.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


def save_forecast(forecast, filename):
    # Ensure forecast is 1-dimensional
    forecast = np.asarray(forecast).flatten()
    pd.DataFrame({"Forecast": forecast}).to_csv(filename, index=False)


def main():
    # ARIMA & SARIMA
    print("\nRunning ARIMA and SARIMA models...")
    arima_plot = os.path.join(output_dir, "arima_forecast.png")
    arima_forecast = arima_sarima.run_model(input_csv, arima_plot, model_type="arima", return_forecast=True)
    save_forecast(arima_forecast, os.path.join(output_dir, "arima_forecast.csv"))


    sarima_plot = os.path.join(output_dir, "sarima_forecast.png")
    sarima_forecast = arima_sarima.run_model(input_csv, sarima_plot, model_type="sarima", return_forecast=True)
    save_forecast(sarima_forecast, os.path.join(output_dir, "sarima_forecast.csv"))


    # LSTM
    print("\nRunning LSTM/GRU model...")
    lstm_plot = os.path.join(output_dir, "lstm_forecast.png")
    lstm_forecast = lstm_gru.run_model(input_csv, lstm_plot, model_type="lstm", sequence_length=30, epochs=50, batch_size=32, return_forecast=True)
    save_forecast(lstm_forecast, os.path.join(output_dir, "lstm_forecast.csv"))


    # HMM
    print("\nRunning HMM model...")
    hmm_plot = os.path.join(output_dir, "hmm_forecast.png")
    hmm_forecast = hmm.run_model(input_csv, hmm_plot, n_states=3, return_forecast=True)
    save_forecast(hmm_forecast, os.path.join(output_dir, "hmm_forecast.csv"))


    # Transformer
    print("\nRunning Transformer Encoder-Decoder model...")
    # Prepare CSV to ensure 'Close' column exists for Transformer
    df = pd.read_csv(input_csv)
    if "Close" not in df.columns and "S&P500" in df.columns:
        df["Close"] = df["S&P500"]
        temp_csv = os.path.join(output_dir, "sp500_temp.csv")
        df.to_csv(temp_csv, index=False)
        transformer_input_csv = temp_csv
    else:
        transformer_input_csv = input_csv

    transformer_plot = os.path.join(output_dir, "transformer_forecast.png")
    transformer_forecast = transformer_forecaster.run_transformer_forecaster(
        transformer_input_csv, transformer_plot, forecast_horizon=30, return_forecast=True)
    save_forecast(transformer_forecast, os.path.join(output_dir, "transformer_forecast.csv"))


    # Ensemble
    print("\nRunning Ensemble model...")
    predictions_dict = {
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast,
        "LSTM": lstm_forecast,
        "HMM": hmm_forecast,
    }
    weights = {"ARIMA": 0.25, "SARIMA": 0.25, "LSTM": 0.25, "HMM": 0.25}
    ensemble_output = os.path.join(output_dir, "ensemble_forecast.png")
    ensemble.run_model(input_csv, ensemble_output, predictions_dict, weights=weights)


    # Other models
    print("\nRunning Anomaly Detection...")
    anomaly_output = os.path.join(output_dir, "anomaly_detection.png")
    anomaly.run_model(input_csv, anomaly_output)


    print("\nRunning Sentiment Analysis...")
    sentiment.run_model(days_before_after=3)


    print("\nRunning RL Trading Agent...")
    rl_agent_output = os.path.join(output_dir, "rl_agent_trading.png")
    rl_agent.train_rl_agent(input_csv, rl_agent_output)


    print("\nRunning GAN model for synthetic stock sequences...")
    gan_output = os.path.join(output_dir, "gan_synthetic.png")
    gans.run_gan(input_csv, gan_output)


    print("\nRunning SHAP & LIME explainers...")
    shap_lime.run_explainers(input_csv, output_dir)


    print("\nAll models completed. Graphs and reports saved in 'outputs/' directory.")


if __name__ == "__main__":
    main()
