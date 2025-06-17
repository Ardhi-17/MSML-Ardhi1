import argparse
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

df = pd.read_csv(args.dataset)
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

for col in X.columns:
    if X[col].dtype == object:
        if col.lower() in ['blood_pressure', 'bp', 'tekanan_darah']:
            X[['BP_sys', 'BP_dia']] = X[col].str.split('/', expand=True).astype(float)
        else:
            X = X.drop(columns=[col])

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # ========== PENTING ========== 
    mlflow.sklearn.log_model(model, "model")  # WAJIB, hasilkan folder model/MLmodel
    # ============================= 
    joblib.dump(model, "model_sleep.joblib")
    mlflow.log_artifact("model_sleep.joblib")
    print("[✓] Model trained & logged as MLflow model + joblib artifact.")

    acc = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", acc)
    print(f"[✓] Test Accuracy: {acc:.4f}")
