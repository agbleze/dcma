FROM python:3.10.14

RUN pip install --upgrade pip

RUN pip install mlflow boto3

EXPOSE 7000

CMD ["python", "-m", "mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db",\
"--default-artifact-root", "s3://mlflow-artifacts", \
"--host", "0.0.0.0", \
"--port", "7000"]