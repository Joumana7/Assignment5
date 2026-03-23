import os
import sys
import mlflow

THRESHOLD = 0.85

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print(f"Using threshold: {THRESHOLD}")
print(f"Model accuracy: {accuracy}")

if accuracy is None:
    print("ERROR: Accuracy not found in MLflow run.")
    sys.exit(1)

accuracy = float(accuracy)

if accuracy < THRESHOLD:
    print(f"Deployment blocked. Accuracy {accuracy} is below threshold {THRESHOLD}.")
    sys.exit(1)

print("Threshold check passed. Proceeding to deployment.")
