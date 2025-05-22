docker run -it --rm -p 7000:7000 \
    -e MLFLOW_S3_ENDPOINT_URL="http://localhost:9000" \
    -e AWS_ACCESS_KEY_ID="minioadmin" \
    -e AWS_SECRET_ACCESS_KEY="minioadmin" \
    agbleze/mlflow:latest 
    # python -m mlflow server \
    # --backend-store-uri sqlite:///mlflow.db \
    # --default-artifact-root s3://mlflow-artifacts \
    # --host 0.0.0.0
