from train import ModelTrainer
import argparse
from utils import read_minio_data
import pandas as pd
from data_ingest import (get_bucket_records, download_from_minio,
                         get_minio_client,
                         get_expected_params_for_func,
                         get_object_name, get_positions,
                         get_variable_position_from_minio_metadata
                         )
import logging
import numpy as np
import json
import os
import functools
from mlflow_wrapper import run_with_mlflow

logging.basicConfig(level=logging.INFO,
                             format="%(asctime)s - %(levelname)s - %(message)s"
                            )

logger = logging.getLogger(__name__)




def dynamic_log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        custom_msg = kwargs.pop("_log_msg", None)
        if not custom_msg:
            pass
        
        
    

def parse_argumments():
    parser = argparse.ArgumentParser(description="Run Model training")
    parser.add_argument("--model_registry", type=str, default="mlflow",
                        help="Model registry to use. Currently only mlflow is supported"
                        )
    parser.add_argument("--read_data_from_minio", action="store_true",
                        help="Whether to read data from MinIO"
                        )
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--scoring", required=True, nargs="+")
    parser.add_argument("--evaluation_metric", required=True)
    parser.add_argument("--save_model_as", required=True)
    parser.add_argument("--cv", default=20)
    parser.add_argument("--model_result_metrics",
                        nargs="+"
                        )
    parser.add_argument("--download_bucket_name", type=str)
    parser.add_argument("--dataset_uid", type=str, required=True)
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
    parser.add_argument("--target_variable", type=str, required=True,
                        help="Target variable name"
                        )
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True,
                        help="MLflow tracking URI"
                        )
    parser.add_argument("--predictor_variables", nargs="+", required=True,
                        help="Predictor variable names"
                        )
    
    
    return parser.parse_args()
    

def main():
    args = parse_argumments()
    logger.info(f"Arguments: {args}")
    train_obj_name = None
    test_obj_name = None
    target_variable = args.target_variable
    predictors = args.predictor_variables
    train_target_position = None
    test_target_position = None
    train_predictor_positions = None
    test_predictor_positions = None
    
    if args.read_data_from_minio or args.upload_output_to_minio:
        minio_client = get_minio_client(args=args)
    if args.read_data_from_minio:
        
        bucket_records = get_bucket_records(bucket_name=args.download_bucket_name,
                                            minio_client=minio_client
                                            )
        
        for bc in bucket_records:
            if (train_obj_name and test_obj_name and train_target_position and 
                test_target_position and train_predictor_positions and
                test_predictor_positions
                ):
                break
            
            if not train_obj_name:
                train_obj_name = get_object_name(bucket_record=bc, dataset_uid=args.dataset_uid, 
                                                 split_type="train"
                                                 )
            if not test_obj_name:
                test_obj_name = get_object_name(bucket_record=bc, dataset_uid=args.dataset_uid, 
                                                split_type="test"
                                                )
            if not train_target_position:
                if isinstance(target_variable, str):
                    target_variable = [target_variable]
                train_target_position = get_variable_position_from_minio_metadata(bucket_record=bc, 
                                                                                  variable=target_variable[0], 
                                                                                    split_type="train", 
                                                                                    dataset_uid=args.dataset_uid,
                                                                                    _log_msg=f"Object name {bc.object_name} Train Target"
                                                                                    )
            if not test_target_position:
                if isinstance(target_variable, str):
                    target_variable = [target_variable]
                test_target_position = get_variable_position_from_minio_metadata(bucket_record=bc, 
                                                             variable=target_variable[0],
                                                              split_type="test", 
                                                              dataset_uid=args.dataset_uid,
                                                              _log_msg=f"Object name {bc.object_name} Test Target"
                                                              )
            if not train_predictor_positions:
                train_predictor_positions = get_variable_position_from_minio_metadata(bucket_record=bc, 
                                                                                      variable=predictors, 
                                                                                    split_type="train", 
                                                                                    dataset_uid=args.dataset_uid,
                                                                                    _log_msg=f"Object name {bc.object_name} Train Predictor{'s' if len(predictors) > 1 else ''}"
                                                                                    )
            if not test_predictor_positions:
                test_predictor_positions = get_variable_position_from_minio_metadata(bucket_record=bc, 
                                                                                     variable=predictors, 
                                                                                    split_type="test", 
                                                                                    dataset_uid=args.dataset_uid,
                                                                                    _log_msg=f"Object name {bc.object_name} Test Predictor{'s' if len(predictors) > 1 else ''}"
                                                                                    )
                
                
        retrieved_bucket_record_objs = {"train object name": train_obj_name, 
                                        "test object name": test_obj_name,
                                        "train target position": train_target_position, 
                                        "test target position": test_target_position, 
                                        "train predictor positions": train_predictor_positions,
                                        "test predictor positions": test_predictor_positions
                                        }
        not_found_bucket_record_objs = [key for key, value in retrieved_bucket_record_objs.items() if value is None]   
        
        if not_found_bucket_record_objs:
            logger.error(f"Could not retrieve the following bucket record objects: {not_found_bucket_record_objs}")
            raise ValueError(f"Could not retrieve the following bucket record objects: {not_found_bucket_record_objs}")
 
        
        if args.include_sample_weight:
            train_class_weight = [bc.metadata.get("class_weight") for bc in bucket_records 
                                  if bc.object_name == train_obj_name
                                ][0]
            
        
        train_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=train_obj_name,
                                        dytpe="npz"
                                        )
        test_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=test_obj_name,
                                        dytpe="npz"
                                        )
    elif not args.read_data_from_minio:
        train_data = np.load(args.local_train_data_path)
        test_data = np.load(args.local_test_data_path)
        
        with open(args.train_preprocessed_metadata_filepath, "rb") as f:
            train_preprocessed_metadata = json.load(f)
        train_class_weight = train_preprocessed_metadata.get("class_weight")
        
        with open(args.test_preprocessed_metadata_filepath, "r") as fp:
            test_preprocessed_metadata = json.load(fp)
        
        train_var_pos_metadata = train_preprocessed_metadata.get("column_positions", {})
        test_var_pos_metadata = test_preprocessed_metadata.get("columns_positions", {})
        
        if not train_var_pos_metadata:
            logger.error(f"Local file {args.local_train_data_path} Train Column positions metadata not found with key 'columns_positions'.")
            raise ValueError(f"Local file {args.local_train_data_path} Train Column positions metadata not found with key 'columns_positions'.")
        
        if not test_var_pos_metadata:
            logger.error(f"Local file {args.local_test_data_path} Test Column positions metadata not found with key 'columns_positions'.")
            raise ValueError(f"Local file {args.local_test_data_path} Test Column positions metadata not found with key 'columns_positions'.")
        
        train_target_position = get_positions(metadata=train_var_pos_metadata, 
                                              variable_names=target_variable,
                                              _log_msg=f"Local file {args.local_train_data_path} Train Target"
                                              )
        test_target_position = get_positions(metadata=test_var_pos_metadata,
                                              variable_names=target_variable,
                                              _log_msg=f"Local file {args.local_test_data_path} Test Target"
                                              )
        train_predictor_positions = get_positions(metadata=train_var_pos_metadata,
                                                  variable_names=predictors,
                                                  _log_msg=f"Local file {args.local_train_data_path} Train Predictor{'s' if len(predictors) > 1 else ''}"
                                                  )
        test_predictor_positions = get_positions(metadata=test_var_pos_metadata,
                                                  variable_names=predictors,
                                                  _log_msg=f"Local file {args.local_test_data_path} Test Predictor{'s' if len(predictors) > 1 else ''}"
                                                  )
        
    #train_npz = train_data.get("preprocessed_data")
    #training_target = train_npz[:, train_target_position]
    #training_predictors = train_npz[:, train_predictor_positions]
    predictor_names = train_data.get("predictor_names")
    logger.info(f"Train predictor names: {predictor_names}")
    if len(predictor_names) == len(predictors):
        training_predictors = train_data.get("predictors") #train_npz[:, train_predictor_positions]
        testing_predictors = test_data.get("predictors")
    else:
        for p in predictors:
            if p not in predictor_names:
                logger.warning(f"Predictor {p} is not a valid predictor name in data.")
                raise ValueError(f"Predictor {p} is not a valid predictor name in data.")
        
        if training_predictors.shape[1] > len(predictors):
            if predictor_names[-1] in predictors:
                pred_index = predictor_names.index(predictor_names[-1])
                trail_train_predictors = training_predictors[:, pred_index:]
                non_trail_pred_index = [predictor_names.index(p) for p in predictors if p not in predictor_names[-1]]
                non_trail_training_predictors = training_predictors[:, non_trail_pred_index]
                training_predictors = np.hstack((non_trail_training_predictors, trail_train_predictors))
                
                trail_test_predictors = testing_predictors[:, pred_index:]
                non_trail_testing_predictors = testing_predictors[:, non_trail_pred_index]
                testing_predictors = np.hstack((non_trail_testing_predictors, trail_test_predictors))
                
            else:
                train_pred_index = [predictor_names.index(p) for p in predictors]
                training_predictors = training_predictors[:, train_pred_index]
                
                testing_predictors = testing_predictors[:, train_pred_index]
  
  
    #training_predictors = train_data.get("predictors")
    training_target = train_data.get(args.target_variable)
    #train_prednms = train_data.get("predictor_names")
    
    
    
    #test_npz = test_data.get("preprocessed_data")
    #testing_target = test_npz[:, test_target_position]
    #testing_predictors = test_npz[:, test_predictor_positions]
    testing_target = test_data.get(args.target_variable)
    testing_predictors = test_data.get("predictors")
    test_prednms = test_data.get("predictor_names")
    logger.info(f"Test predictor names: {test_prednms}")
    
    if args.include_sample_weight:
        train_sample_weight = np.array([train_class_weight[int(label)] for label in training_target])
    else:
        train_sample_weight = None
    logger.info(f"Train predictors shape: {training_predictors.shape}")
    logger.info(f"Test predictors shape: {testing_predictors.shape}")
    #logger.info(f"Positions of predictors: {train_predictor_positions}")
    #exit()
    trainer = ModelTrainer(training_predictors=training_predictors,
                            training_target=training_target,
                            testing_predictors=testing_predictors,
                            testing_target=testing_target,
                            sample_weight=train_sample_weight,
                            model_type=args.model_type,
                            #model_registry=args.model_registry,
                            )
    run_with_mlflow(trainer=trainer, run_params=vars(args),
                    tracking_uri=args.mlflow_tracking_uri
                    )

if __name__ == "__main__":
    main()






