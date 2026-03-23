import os
import sys
import mlflow

THRESHOLD = 0.85

def main():
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("ERROR: model_info.txt not found.")
        sys.exit(1)

    if not run_id:
        print("ERROR: Run ID is empty.")
        sys.exit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not retrieve run '{run_id}': {e}")
        sys.exit(1)

    accuracy = run.data.metrics.get("accuracy")

    print(f"Using threshold: {THRESHOLD}")
    print(f"Run ID: {run_id}")
    print(f"Model accuracy: {accuracy}")

    if accuracy is None:
        print("ERROR: No 'accuracy' metric found.")
        sys.exit(1)

    accuracy = float(accuracy)

    if accuracy < THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}. Blocking deployment.")
        sys.exit(1)

    print(f"PASS: Accuracy {accuracy:.4f} meets the threshold. Proceeding to deployment.")
    sys.exit(0)

if __name__ == "__main__":
    main()
