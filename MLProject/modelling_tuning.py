import argparse
import mlflow
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.dataset)

# Drop kolom yang tidak numerik
# Kamu bisa ubah sesuai kolom yang ada
drop_cols = ['Blood Pressure', 'Occupation', 'Gender', 'BMI Category']  # tambah jika perlu
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Pisahkan fitur & target
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Save & log model
    joblib.dump(model, "model_sleep.joblib")
    mlflow.log_artifact("model_sleep.joblib")
    print("[✓] Model trained & logged as artifact.")
