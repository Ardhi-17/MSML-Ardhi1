import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- Argument parsing untuk MLflow Project ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

# --- (Opsional) Set experiment name, tidak start_run! ---
mlflow.set_experiment("Sleep Disorder Tuning (CI)")

# --- Aktifkan autolog, supaya model, params, dsb otomatis ke mlruns/0 ---
mlflow.sklearn.autolog()

# --- Load dataset ---
df = pd.read_csv(args.dataset)
features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps'
]
target = 'Sleep Disorder'
X = df[features]
y = df[target]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Standardisasi fitur numerik ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_sleep.joblib')  # bisa juga log manual jika mau

# --- Hitung class weight (dengan np.unique) ---
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict = dict(zip(classes, class_weights))

# --- GridSearchCV untuk hyperparameter tuning ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5, 7],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(
    random_state=42,
    class_weight=cw_dict
)
search = GridSearchCV(rf, param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=2)
search.fit(X_train_scaled, y_train)

# --- Save best model & scaler secara manual (opsional) ---
best_model = search.best_estimator_
joblib.dump(best_model, 'model_sleep_tuned.joblib')

# --- Log params & metric best secara manual (autolog tetap jalan) ---
mlflow.log_params(search.best_params_)
mlflow.log_metric("best_cv_score", search.best_score_)

print(f"[✓] Model tuning selesai. Best params: {search.best_params_}, Akurasi: {search.best_score_:.4f}")
