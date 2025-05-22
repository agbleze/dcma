import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.DEBUG,
                     format="%(asctime)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)

def run_with_mlflow(trainer, run_params, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as run:
        mlflow.log_params(run_params)
        
        trainer.run_model_training_pipeline(cv=20)
        mlflow.log_metrics("train_score", trainer.train_score)
        mlflow.log_metrics("test_score", trainer.test_score)
        
        mlflow.sklearn.log_model(trainer.estimator, artifact_path="model")
        
        logger.info(f"Completed model logging")
        