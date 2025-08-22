import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def run_model(input_csv, output_path, predictions_dict, weights=None):
    """
    Run ensemble forecast by combining multiple model predictions.

    Parameters:
    - input_csv: path to the original CSV with Date and Price columns
    - output_path: path to save the ensemble graph
    - predictions_dict: dictionary with model_name: np.ndarray or pd.Series(predictions)
    - weights: dictionary with model_name: weight (sum should be 1.0)
    """

    # -------------------------------
    # Step 1: Load original data
    # -------------------------------
    df = pd.read_csv(input_csv)
    if 'Date' not in df.columns or df.columns[1] not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Price' columns")

    price_col = df.columns[1]  # handle S&P500 or Price
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date', price_col]).sort_values('Date')
    df = df.set_index('Date')

    # -------------------------------
    # Step 2: Prepare weights
    # -------------------------------
    if weights is None:
        # default: equal weights
        weights = {model: 1/len(predictions_dict) for model in predictions_dict}

    # -------------------------------
    # Step 3: Compute ensemble forecast
    # -------------------------------
    ensemble_forecast = None
    for model_name, series in predictions_dict.items():
        # Convert NumPy array to pandas Series if needed
        if isinstance(series, np.ndarray):
            forecast_dates = pd.date_range(df.index[-1], periods=len(series)+1, freq='B')[1:]
            series = pd.Series(series.flatten(), index=forecast_dates)
            predictions_dict[model_name] = series  # update dictionary with Series

        weighted_series = series * weights.get(model_name, 0)
        if ensemble_forecast is None:
            ensemble_forecast = weighted_series
        else:
            ensemble_forecast = ensemble_forecast.add(weighted_series, fill_value=0)

    # -------------------------------
    # Step 4: Plot results
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(df[price_col], label="Actual", color="blue")

    # Plot individual model predictions
    for model_name, series in predictions_dict.items():
        plt.plot(series.index, series, label=f"{model_name} Forecast", alpha=0.6)

    # Plot ensemble
    plt.plot(ensemble_forecast.index, ensemble_forecast, label="Ensemble Forecast", color="red", linewidth=2)

    plt.xlabel("Date")
    plt.ylabel(price_col)
    plt.title("Ensemble Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # -------------------------------
    # Step 5: Print summary table
    # -------------------------------
    table_data = []
    for model_name, series in predictions_dict.items():
        table_data.append([model_name, series.iloc[-1]])
    table_data.append(["Ensemble", ensemble_forecast.iloc[-1]])

    print("\nENSEMBLE FORECAST SUMMARY")
    print(tabulate(table_data, headers=["Model", "Last Forecast Value"], tablefmt="grid"))

    # -------------------------------
    # Step 6: Save report as TXT
    # -------------------------------
    report_path = os.path.splitext(output_path)[0] + "_ensemble_report.txt"
    with open(report_path, "w") as f:
        f.write("ENSEMBLE MODEL REPORT\n")
        f.write("="*50 + "\n\n")
        f.write("Weights used:\n")
        for model_name, w in weights.items():
            f.write(f"- {model_name}: {w}\n")
        f.write("\n")
        f.write(tabulate(table_data, headers=["Model", "Last Forecast Value"], tablefmt="grid"))

    print(f"\nEnsemble graph saved at: {output_path}")
    print(f"Ensemble report saved at: {report_path}")
