import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# CLI arg
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

df = pd.read_csv(args.dataset)

features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps'
]
target = 'Sleep Disorder'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_sleep.joblib')

# Class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict = dict(zip(classes, class_weights))

# Hyperparam tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 8, None],
    'min_samples_split': [2, 5]
}
clf = RandomForestClassifier(random_state=42, class_weight=cw_dict)
grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_params = grid.best_params_
best_model = grid.best_estimator_
joblib.dump(best_model, 'model_sleep_tuned.joblib')

# Metrics
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

with mlflow.start_run():
    # Log MLflow model (for serving/docker)
    mlflow.sklearn.log_model(best_model, "model")
    # Log artifacts
    mlflow.log_artifact("scaler_sleep.joblib")
    mlflow.log_artifact("model_sleep_tuned.joblib")
    mlflow.log_artifact("confusion_matrix.png")
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    # Log best hyperparameters
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    print(f"[✓] Model tuning selesai. Best params: {best_params}, Akurasi: {acc:.4f}")
