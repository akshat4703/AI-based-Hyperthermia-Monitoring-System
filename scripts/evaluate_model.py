# scripts/evaluate_model.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate():
    model = joblib.load(os.path.join(MODEL_DIR, "hyperthermia_temp_model.pkl"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))['Temperature']

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.6f}\nR2: {r2:.6f}\n")

    # Plot Predicted vs Actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Predicted vs Actual Temperature")
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "pred_vs_actual.png")
    plt.savefig(out_png, dpi=150)
    print("Plot saved to", out_png)

if __name__ == "__main__":
    evaluate()
