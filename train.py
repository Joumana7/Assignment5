import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- MLflow setup ---
# MLFLOW_TRACKING_URI is injected via GitHub Secret in CI,
# or set locally before running this script.
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("assignment5-classifier")

# --- Data ---
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# --- Train ---
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    print(f"Run ID : {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    # Write the Run ID to a file so the deploy job can read it
    with open("model_info.txt", "w") as f:
        f.write(run_id)

print("Training complete. model_info.txt written.")
