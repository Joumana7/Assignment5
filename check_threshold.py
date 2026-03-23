"""
check_threshold.py
------------------
Reads the MLflow Run ID from model_info.txt, fetches the logged
'accuracy' metric and fails (exit 1) if it is below THRESHOLD.

Usage:
    python check_threshold.py           # reads model_info.txt in cwd
    python check_threshold.py <run_id>  # pass run id directly
"""

import os
import sys
import mlflow

THRESHOLD = 0.50


def main():
    # --- Get Run ID ---
    if len(sys.argv) > 1:
        run_id = sys.argv[1].strip()
    else:
        try:
            with open("model_info.txt", "r") as f:
                run_id = f.read().strip()
        except FileNotFoundError:
            print("ERROR: model_info.txt not found and no run_id argument given.")
            sys.exit(1)

    if not run_id:
        print("ERROR: Run ID is empty.")
        sys.exit(1)

    # --- Connect to MLflow ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not retrieve run '{run_id}': {e}")
        sys.exit(1)

    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print(f"ERROR: No 'accuracy' metric found for run '{run_id}'.")
        sys.exit(1)

    print(f"Run ID  : {run_id}")
    print(f"Accuracy: {accuracy:.4f}  (threshold: {THRESHOLD})")

    if accuracy < THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}. Blocking deployment.")
        sys.exit(1)

    print(f"PASS: Accuracy {accuracy:.4f} meets the threshold. Proceeding to deployment.")
    sys.exit(0)


if __name__ == "__main__":
    main()
