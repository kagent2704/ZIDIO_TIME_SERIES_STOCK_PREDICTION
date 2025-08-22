import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# -------------------------------
# Config
# -------------------------------
INPUT_CSV = "data/sp500.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Run SHAP & LIME
# -------------------------------
def run_explainers(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR):
    # Load data
    df = pd.read_csv(input_csv)
    
    # Use S&P500 lag features for demonstration
    df['Lag1'] = df['S&P500'].shift(1)
    df['Lag2'] = df['S&P500'].shift(2)
    df['Lag3'] = df['S&P500'].shift(3)
    df = df.dropna()
    
    X = df[['Lag1','Lag2','Lag3']].values
    y = df['S&P500'].values
    
    # Train simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # -------------------------------
    # SHAP
    # -------------------------------
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)
    
    shap_plot_path = os.path.join(output_dir, "shap_summary.png")
    shap.summary_plot(shap_values, X_test, feature_names=['Lag1','Lag2','Lag3'], show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved: {shap_plot_path}")
    
    # -------------------------------
    # LIME
    # -------------------------------
    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=['Lag1','Lag2','Lag3'],
        mode='regression'
    )
    
    # Pick a random test instance
    i = np.random.randint(0, X_test.shape[0])
    exp = lime_explainer.explain_instance(X_test[i], model.predict, num_features=3)
    
    lime_plot_path = os.path.join(output_dir, "lime_explanation.png")
    fig = exp.as_pyplot_figure()
    fig.savefig(lime_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"LIME explanation saved: {lime_plot_path}")

