# Base image
FROM python:3.10-slim

# Build-time argument: the MLflow Run ID of the model to serve
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# Install minimal dependencies
RUN pip install --no-cache-dir mlflow scikit-learn

WORKDIR /app

# Simulate downloading the model from MLflow
# In production, replace the echo with:
#   mlflow artifacts download -r ${RUN_ID} -d /app/model
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "${RUN_ID}" > /app/run_id.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: print which model is loaded
CMD ["sh", "-c", "echo 'Serving model for Run ID:' $(cat /app/run_id.txt)"]
