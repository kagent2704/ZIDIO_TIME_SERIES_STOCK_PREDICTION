# models/transformer_forecaster.py

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Transformer definition (same as before) ---
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # predict last step


def run_transformer_forecaster(input_csv, output_plot, forecast_horizon=30, return_forecast=True):
    print("Running Transformer-based stock forecaster...")

    # Load dataset
    df = pd.read_csv(input_csv)
    prices = df["Close"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences
    seq_len = 30
    X, y = [], []
    for i in range(len(prices_scaled) - seq_len):
        X.append(prices_scaled[i:i+seq_len])
        y.append(prices_scaled[i+seq_len])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Model
    model = TransformerForecaster(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train (light for demo)
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3, Loss: {loss.item():.6f}")

    # Generate forecasts
    last_seq = torch.tensor(prices_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    preds = []
    for _ in range(forecast_horizon):
        with torch.no_grad():
            pred = model(last_seq).item()
        preds.append(pred)
        new_val = torch.tensor([[[pred]]])
        last_seq = torch.cat([last_seq[:, 1:], new_val], dim=1)

    # Inverse scale
    preds = scaler.inverse_transform([[p] for p in preds]).flatten()

    # Save plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(prices)), prices, label="Actual Prices")
    plt.plot(range(len(prices), len(prices)+forecast_horizon), preds, label="Forecast", color="red")
    plt.legend()
    plt.title("Transformer Stock Price Forecast")
    plt.savefig(output_plot)
    plt.close()

    # Save CSV
    forecast_csv = os.path.join(os.path.dirname(output_plot), "transformer_forecast.csv")
    pd.DataFrame({"Day": range(1, forecast_horizon+1), "Forecast": preds}).to_csv(forecast_csv, index=False)
    print(f"Forecast saved to {forecast_csv}")

    if return_forecast:
        return preds
