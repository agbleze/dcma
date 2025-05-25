from train import ModelTrainer
import argparse
from utils import read_minio_data
import pandas as pd
from data_ingest import (get_bucket_records, download_from_minio,
                         get_minio_client,
                         get_expected_params_for_func
                         )
import logging
import numpy as np
import json
import os
import functools

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
        
        
    
def get_positions(metadata: dict, variable_names: list, **kwargs):
    custom_msg = kwargs.pop("_log_msg", None)
    variables_not_found = [pred for pred in variable_names if pred not in metadata]
    if variables_not_found:
        logger.error(f"{custom_msg if custom_msg else ''} {variables_not_found} not found in metadata.")
        raise ValueError(f"{variables_not_found} not found in metadata.")
    var_positions = [metadata[var] for var in variable_names]
    logger.info(f"{custom_msg if custom_msg else ''} '{variable_names}' {'are' if len(variable_names) > 1 else 'is'} at position{f's {var_positions} (indices)' if len(var_positions)>1 else f' {var_positions} (index)'}")
    return var_positions


def get_object_name(bucket_record, dataset_uid, split_type):
    if (bucket_record.metadata.get("uid") == dataset_uid) and (bucket_record.metadata["split_type"] == split_type):
        obj_name = split_type.object_name
        if not obj_name:
            logger.info(f"Retrieved {split_type} object name: {obj_name}")
    else:
        obj_name = None
    return obj_name

def get_variable_position_from_minio_metadata(bucket_record, variable: list, 
                                              split_type: str, 
                                              dataset_uid: str,
                                              **kwargs
                                              ):
    _log_msg = kwargs.pop("_log_msg", None)
    if (bucket_record.get("uid") == dataset_uid) and (bucket_record["split_type"] == split_type):
        var_position_metadata = bucket_record.metadata.get("column_positions", {})
        if isinstance(variable, str):
            variable = [variable]
        var_position = get_positions(metadata=var_position_metadata, variable_names=variable,
                                     _log_msg=_log_msg
                                     )
    else:
        var_position = None
    return var_position

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
    parser.add_argument("--dataset_uid", type=str)
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
    
    
    return parser.parse_args()
    

def main():
    args = parse_argumments()
    logger.info(f"Arguments: {args}")
    print(f"args type: {type(args)}")
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
        #print(bucket_records[0].metadata["uid"])
        # train_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
        #                     and (bc.metadata["split_type"] == "train")
        #                 ]
        
        
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
                                                                                  variable=target_variable, 
                                                                                    split_type="train", 
                                                                                    dataset_uid=args.dataset_uid,
                                                                                    _log_msg=f"Object name {bc.object_name} Train Target"
                                                                                    )
            if not test_target_position:
                if isinstance(target_variable, str):
                    target_variable = [target_variable]
                test_target_position = get_variable_position_from_minio_metadata(bucket_record=bc, 
                                                             variable=target_variable,
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
        
        
                
            # if (bc.metadata.get("uid") == args.dataset_uid) and (bc.metadata["split_type"] == "train"):
            #     train_obj_name = bc.object_name
            #     logger.info(f"Retrieved train object name: {train_obj_name}")
            #     col_position = bc.metadata.get("column_positions", {})
            #     if not col_position:
            #         logger.error(f"Column positions metadata not found for Train data object {train_obj_name}.")
            #         raise ValueError(f"Column positions metadata not found for Train data object {train_obj_name}.")
                    
            #     if target_variable in col_position:
            #         train_target_position = col_position[target_variable]
            #         logger.info(f"Target variable '{target_variable}' in Train data is at position: {train_target_position} (index)")
            #     else:
            #         logger.error(f"Target variable '{target_variable}' not found in Train data column positions metadata.")
            #         raise ValueError(f"Target variable '{target_variable}' not found in Train data column positions metadata.")
                
            #     predictors_not_found = [pred for pred in predictors if pred not in col_position]
            #     if predictors_not_found:
            #         logger.error(f"Predictor variables {predictors_not_found} not found in Train data column positions metadata.")
            #         raise ValueError(f"Predictor variables {predictors_not_found} not found in Train data column positions metadata.")
            #     train_predictor_positions = [col_position[predictor] for predictor in predictors]
            #     logger.info(f"Predictor variables '{predictors}' in Train data are at positions: {train_predictor_positions} (indices)")
            #     #break
            
            # if (bc.metadata.get("uid") == args.dataset_uid) and (bc.metadata["split_type"] == "test"):
            #     test_obj_name = bc.object_name
            #     logger.info(f"Retrieved test object name: {test_obj_name}")

            #     col_position = bc.metadata.get("column_positions", {})
            #     if not col_position:
            #         logger.error(f"Column positions metadata not found in Test data for object {test_obj_name}.")
            #         raise ValueError(f"Column positions metadata not found in Test data for object {test_obj_name}.")
            #     if target_variable in col_position:
            #         test_target_position = col_position[target_variable]
            #         logger.info(f"Target variable '{target_variable}' in Test data is at position: {test_target_position} (index)")
            #     else:
            #         logger.error(f"Target variable '{target_variable}' not found in Test data column positions metadata.")
            #         raise ValueError(f"Target variable '{target_variable}' not found in Test data column positions metadata.")
            #     predictors_not_found = [pred for pred in predictors if pred not in col_position]
            #     if predictors_not_found:
            #         logger.error(f"Predictor variables {predictors_not_found} not found in Test data column positions metadata.")
            #         raise ValueError(f"Predictor variables {predictors_not_found} not found in Test data column positions metadata.")
            #     test_predictor_positions = [col_position[predictor] for predictor in predictors]
            #     logger.info(f"Predictor variables '{predictors}' in Test data are at positions: {test_predictor_positions} (indices)")
            # break
            
        
        # test_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
        #                     and (bc.metadata["split_type"] == "test")
        #                 ]
        # logger.info(f"Retrieved test object name: {test_obj_name}")
        
        
        
        if args.include_sample_weight:
            train_class_weight = [bc.metadata.get("class_weight") for bc in bucket_records if bc.object_name == train_obj_name[0]
                                ][0]
            
        
        train_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=train_obj_name[0],
                                        dytpe="npz"
                                        )
        test_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=test_obj_name[0],
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
        
        # add dynamic logging decorator with message eg. Train data metadata  for column positions
        
        
        
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
        
    train_npz = train_data.get("preprocessed_data")
    training_target = train_npz[:, train_target_position]
    training_predictors = train_npz[:, train_predictor_positions]
    
    test_npz = test_data.get("preprocessed_data")
    testing_target = test_npz[:, test_target_position]
    testing_predictors = test_npz[:, test_predictor_positions]
    
    if args.include_sample_weight:
        train_sample_weight = np.array([train_class_weight[int(label)] for label in training_target])
    else:
        train_sample_weight = None
    print("############# running model  ##########")
    trainer = ModelTrainer(training_predictors=training_predictors,
                            training_target=training_target,
                            testing_predictors=testing_predictors,
                            testing_target=testing_target,
                            sample_weight=train_sample_weight,
                            model_type=args.model_type,
                            #model_registry=args.model_registry,
                            )
    trainer.run_model_training_pipeline(cv=20, 
                                        scoring=args.scoring, #['accuracy', "precision", "recall", "f1"],
                                        model_result_metrics=args.model_result_metrics,
                                        
                                        # ['test_accuracy',  'train_accuracy',
                                        #                       'test_precision', 'train_precision',
                                        #                       'test_recall', 'train_recall',
                                        #                       'test_f1', 'train_f1'
                                        #                       ],
                                        evaluation_metric=args.evaluation_metric, #"accuracy_score",
                                        save_model_as=args.save_model_as, #"conversion_proba.model",
                                        save_dir="model_store"
                                        )
    

if __name__ == "__main__":
    main()






