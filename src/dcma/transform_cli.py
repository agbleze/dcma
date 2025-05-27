from transform import transform_data_with_conversion
import argparse
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Run Data transformation")
    parser.add_argument("--conversion_classifier_path", required=True)
    parser.add_argument("--read_data_from_minio",
                        type=bool
                        )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--predictors", required=True)
    parser.add_argument("--upload_output_to_minio", action="store_true")
    parser.add_argument("--access_key_env_name", default="MINIO_ACCESS_KEY", help="Env var name for MinIO access key")
    parser.add_argument("--access_secret_env_name", default="MINIO_SECRET_KEY", help="Env var name for MinIO secret key")
    parser.add_argument("--minio_server_url_env_name", default="MINIO_SERVER_URL", help="Env var name for MinIO server URL")
    parser.add_argument("--minio_endpoint_is_secured", action="store_true",
                        help="Whether the Minio endpoint url is secured"
                        )
    parser.add_argument("--local_train_data_path", type=str)
    parser.add_argument("--local_test_data_path", type=str)
    parser.add_argument("--train_preprocessed_metadata_filepath", type=str)
    parser.add_argument("--test_preprocessed_metadata_filepath", type=str)
    parser.add_argument("--include_sample_weight", action="store_true",
                        help="Whether to include sample weight in the training"
                        )
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True,
                        help="MLflow tracking URI"
                        )
    parser.add_argument("--predictor_variables", nargs="+", required=True,
                        help="Predictor variable names"
                        )
    
    
    args = parser.parse_args()
    
    train_cpa = transform_data_with_conversion(data=train_cpa, 
                                               variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )
    test_cpa = transform_data_with_conversion(data=test_cpa, variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )
    



