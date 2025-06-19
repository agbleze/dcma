import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
from data_ingest import get_expected_params_for_func
import os
import uuid

mlflow.enable_system_metrics_logging()
logging.basicConfig(level=logging.DEBUG,
                     format="%(asctime)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)

def run_with_mlflow(trainer, run_params, tracking_uri):
    AWS_ACCESS_KEY_ID = os.getenv(key=run_params["access_key_env_name"])
    AWS_SECRET_ACCESS_KEY = os.getenv(key=run_params["access_secret_env_name"])
    #MLFLOW_S3_ENDPOINT_URL = os.getenv(key=run_params["minio_server_url_env_name"])
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_suffix = str(uuid.uuid4())
    experiment_id = mlflow.create_experiment(name=f"conversion_classifier_experiment_{experiment_suffix}",
                                            artifact_location="s3://mlflow-artifacts"
                                            )
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run() as run:
        
        logger.info(f"Started MLflow run with experiment_id: {experiment_id}")
        logger.info(f"Run ID: {run.info.run_id}")
        
        mlflow.log_params(run_params)
        run_model_expected_params = get_expected_params_for_func(func=trainer.run_model_training_pipeline,
                                                                 **run_params
                                                                 )
        trainer.run_model_training_pipeline(**run_model_expected_params)
        
        mlflow.log_metric("train_score", trainer.train_score)
        mlflow.log_metric("test_score", trainer.test_score)
        #mlflow.log_metrics(trainer.cv_metrics)
        
        signature = infer_signature(model_input=trainer.training_predictors,
                                    model_output=trainer.estimator.predict(trainer.training_predictors)
                                    )
       # mlflow.register_model()
        
        mlflow.sklearn.log_model(trainer.estimator, artifact_path="model",
                                 signature=signature,
                                 registered_model_name="conversion_classifier",
                                 input_example=trainer.training_predictors
                                 )
        
        logger.info(f"Completed model logging")
        