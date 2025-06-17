import argparse
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Parsing argumen dari MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path ke dataset preprocessing (csv)")
args = parser.parse_args()

# Tracking MLflow experiment
mlflow.set_experiment("Sleep Disorder Tuning (CI)")

# Load dataset
df = pd.read_csv(args.dataset)

features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps'
]
target = 'Sleep Disorder'
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_sleep.joblib')

# Class weight
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict = dict(zip(classes, class_weights))

# Hyperparameter tuning
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

# Manual logging MLflow
with mlflow.start_run():
    # Log params
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    # Log metrik utama & tambahan
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix sebagai artefak
    cm = confusion_matrix(y_test, y_pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log model & scaler
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact("scaler_sleep.joblib")

    print(f"[✓] Model tuning selesai. Best params: {best_params}, Akurasi: {acc:.4f}")

print("[INFO] Tuning & logging selesai. Cek MLflow UI untuk hasil run.")
