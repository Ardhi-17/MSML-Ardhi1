import argparse
import mlflow
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.dataset)
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
    
    # Generate & log confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("conf_matrix.png")
    plt.close()
    mlflow.log_artifact("conf_matrix.png")
    
    print("[✓] Model trained, logged, and confusion matrix saved.")
