import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# MLflow setup
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("assignment5-classifier")

# Data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train and log
with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    with open("model_info.txt", "w") as f:
        f.write(run_id)

print("Training complete. model_info.txt written.")
