import argparse
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Parsing argumen dari MLProject/CI/CD
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

# --- Load data
df = pd.read_csv(args.dataset)

# --- Pastikan semua kolom input numerik ---
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Tangani kolom string (misal, 'Blood_Pressure' dengan format 130/85)
for col in X.columns:
    if X[col].dtype == object:
        # Deteksi kolom tekanan darah / blood pressure
        if col.lower() in ['blood_pressure', 'bp', 'tekanan_darah']:
            # Split jadi dua kolom numerik
            X[['BP_sys', 'BP_dia']] = X[col].str.split('/', expand=True).astype(float)
        else:
            X = X.drop(columns=[col])

# Pastikan hanya numerik
X = X.select_dtypes(include=[np.number])

# --- Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Save & log model as MLflow model (WAJIB untuk build-docker)
    mlflow.sklearn.log_model(model, "model")  # <-- Tambahan WAJIB
    # Save & log joblib model juga (opsional, bonus artefak manual)
    joblib.dump(model, "model_sleep.joblib")
    mlflow.log_artifact("model_sleep.joblib")
    print("[✓] Model trained & logged as MLflow model + joblib artifact.")

    # Log metrics (optional)
    acc = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", acc)
    print(f"[✓] Test Accuracy: {acc:.4f}")
