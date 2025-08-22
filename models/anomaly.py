import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def run_model(input_csv, output_path, column=None, z_thresh=3, if_contamination=0.01):
    """
    Detect anomalies using Statistical method (z-score) and Isolation Forest.

    Parameters:
    - input_csv: path to CSV
    - output_path: path to save anomaly plot
    - column: name of numeric column to analyze (if None, auto-pick first numeric)
    - z_thresh: z-score threshold for statistical anomalies
    - if_contamination: contamination fraction for Isolation Forest
    """
    # -------------------------------
    # Step 1: Load data
    # -------------------------------
    df = pd.read_csv(input_csv)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df['Date'] = pd.RangeIndex(start=0, stop=len(df), step=1)
    
    # Pick numeric column
    if column is None:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric column found in CSV")
        column = numeric_cols[0]

    series = df[column]

    # -------------------------------
    # Step 2: Statistical anomalies
    # -------------------------------
    mean = series.mean()
    std = series.std()
    df['Stat_Anomaly'] = ((series - mean).abs() > z_thresh * std).astype(int)

    # -------------------------------
    # Step 3: Isolation Forest
    # -------------------------------
    iso_forest = IsolationForest(contamination=if_contamination, random_state=42)
    iso_pred = iso_forest.fit_predict(series.values.reshape(-1,1))
    df['IF_Anomaly'] = (iso_pred == -1).astype(int)

    # -------------------------------
    # Step 4: Combined anomalies
    # -------------------------------
    df['Combined_Anomaly'] = ((df['Stat_Anomaly'] + df['IF_Anomaly']) > 0).astype(int)

    # -------------------------------
    # Step 5: Plot anomalies
    # -------------------------------
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], series, label='Value', color='blue')
    plt.scatter(df['Date'][df['Stat_Anomaly']==1], series[df['Stat_Anomaly']==1], 
                color='orange', label='Statistical Anomaly', marker='o')
    plt.scatter(df['Date'][df['IF_Anomaly']==1], series[df['IF_Anomaly']==1], 
                color='green', label='Isolation Forest Anomaly', marker='x')
    plt.scatter(df['Date'][df['Combined_Anomaly']==1], series[df['Combined_Anomaly']==1], 
                color='red', label='Combined Anomaly', marker='D')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title('Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # -------------------------------
    # Step 6: Save CSV report
    # -------------------------------
    report_path = os.path.splitext(output_path)[0] + "_anomalies.csv"
    df.to_csv(report_path, index=False)
    print(f"Anomaly plot saved at: {output_path}")
    print(f"Anomaly CSV saved at: {report_path}")

    return df[['Date', column, 'Stat_Anomaly', 'IF_Anomaly', 'Combined_Anomaly']]

