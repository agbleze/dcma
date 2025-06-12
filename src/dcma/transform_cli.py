from transform import transform_data_with_conversion
import argparse
import pandas as pd
import mlflow
from mlflow.models import Model
import os
from data_ingest import (get_bucket_records, download_from_minio,
                         get_minio_client,
                         get_expected_params_for_func,
                         get_object_name,
                         get_positions,
                         ObjectToPersistData,
                         upload_to_minio,
                         get_variable_position_from_minio_metadata
                         )
import numpy as np
import logging
import io
import uuid
from copy import deepcopy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Data transformation")
    parser.add_argument("--read_data_from_minio", action="store_true",
                        help="Whether to read data from MinIO"
                        )
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
    parser.add_argument("--dataset_uid", type=str, required=True,
                        help="Unique identifier for the dataset"
                        )
    parser.add_argument("--model_uri", type=str, required=True,
                        help="Model URI"
                        )
    parser.add_argument("--predictor_variables", nargs="+", required=True,
                        help="Predictor variable names"
                        )
    parser.add_argument("--download_bucket_name", type=str)
    parser.add_argument("--augment_bucket_name", type=str,
                        help="Bucket name to upload augmented data"
                        )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    predictors = args.predictor_variables
    augment_dataset_uid = str(uuid.uuid4())
    logger.info(f"Predictor variables: {predictors}")
    train_obj_name = None
    test_obj_name = None
    train_predictor_positions = None
    test_predictor_positions = None
    if args.read_data_from_minio:
        minio_client = get_minio_client(args=args)
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(args.access_key_env_name)
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(args.access_secret_env_name)
    pyfunc_model = mlflow.sklearn.load_model(args.model_uri)
    if args.read_data_from_minio:
        logger.info(f"Reading data from MinIO bucket: {args.download_bucket_name}")
        bucket_records = get_bucket_records(bucket_name=args.download_bucket_name,
                                            minio_client=minio_client
                                            )
        
        for bc in bucket_records:
            if (train_obj_name and test_obj_name and 
                 train_predictor_positions and
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
                
            logger.info("Starting to get predictor positions from MinIO metadata")
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
        
        logger.info(f"Completed downloading train and test data from MinIO")
        
        
        
        retrieved_bucket_record_objs = {"train object name": train_obj_name, 
                                        "test object name": test_obj_name,
                                        "train predictor positions": train_predictor_positions,
                                        "test predictor positions": test_predictor_positions
                                        }
        not_found_bucket_record_objs = [key for key, value in retrieved_bucket_record_objs.items() if value is None]   
        
        if not_found_bucket_record_objs:
            logger.error(f"Could not retrieve the following bucket record objects: {not_found_bucket_record_objs}")
            raise ValueError(f"Could not retrieve the following bucket record objects: {not_found_bucket_record_objs}")
       
        logger.info("Completed getting predictor positions from MinIO metadata")
    

    train_npz = train_data.get("preprocessed_data")
    train_cpa = train_data.get("cpa")
    
    
    logger.info(f"Train data shape: {train_npz.shape}")
    logger.info(f"Train Predictor positions: {train_predictor_positions}")
    
    train_predictor_names = train_data.get("predictor_names")
    training_predictors = train_data.get("predictors") #train_npz[:, train_predictor_positions]
    
    # if len(predictor_names) == len(predictors):
    #     training_predictors = train_data.get("predictors") #train_npz[:, train_predictor_positions]
    # else:
    #     for p in predictors:
    #         if p not in predictor_names:
    #             logger.warning(f"Predictor {p} is not a valid predictor name in data.")
        
    #     if training_predictors.shape[1] > len(predictors):
    #         if predictor_names[-1] in predictors:
    #             pred_index = predictor_names.index(predictor_names[-1])
    #             tailing_predictors = training_predictors[:, pred_index:]
            
    
    test_npz = test_data.get("preprocessed_data")
    test_cpa = test_data.get("cpa")
    logger.info(f"Test data shape: {test_npz.shape}")
    logger.info(f"Test Predictor positions: {test_predictor_positions}")
    testing_predictors = test_data.get("predictors") #test_npz[:, test_predictor_positions]
    test_predictor_names = test_data.get("predictor_names")
    logger.info(f"Training predictors shape: {training_predictors.shape}")
    logger.info(f"Testing predictors shape: {testing_predictors.shape}")
    
    train_proba = pyfunc_model.predict_proba(training_predictors)[:, 1]
    logger.info(f"Train prediction shape: {train_proba.shape}")
    logger.info(f"Train prediction: {train_proba}")
    
    train_proba_single_col = train_proba.reshape(-1,1)
    test_proba = pyfunc_model.predict_proba(testing_predictors)[:, 1]
    test_proba_single_col = test_proba.reshape(-1,1)
    
    augmented_train_data = np.hstack([training_predictors, train_proba_single_col])
    augmented_test_data = np.hstack([testing_predictors, test_proba_single_col])
    print(f"test probabilities: {test_proba}")
    print(f"type of test probabilities: {type(test_proba)}")
    logger.info(f"Augmented train data shape: {augmented_train_data.shape}")
    logger.info(f"Augmented test data shape: {augmented_test_data.shape}")
    train_predictor_names = np.append(train_predictor_names, "conversion_probability")
    test_predictor_names = np.append(test_predictor_names, "conversion_probability")
    logger.info(f"Train predictor names: {train_predictor_names}")
    logger.info(f"Test predictor names: {test_predictor_names}")
    
    artifacts_list = []
    augment_train_filename = f"train_{augment_dataset_uid}.npz"
    augment_test_filename = f"test_{augment_dataset_uid}.npz"
    if args.upload_output_to_minio:
        augment_train_buffer = io.BytesIO()
        np.savez_compressed(file=augment_train_buffer, 
                            predictors=augmented_train_data,
                            predictor_names=np.array(train_predictor_names),
                            cpa=train_cpa,
                            )
        augment_train_buffer.seek(0)
        
        augment_test_buffer = io.BytesIO()
        np.savez_compressed(file=augment_test_buffer, 
                            predictors=augmented_test_data,
                            predictor_names=np.array(test_predictor_names),
                            cpa=test_cpa
                            )
        augment_test_buffer.seek(0)
        args_dict = vars(args)
        args_dict["uid"] = augment_dataset_uid
        args_dict["parent_uid"] = args.dataset_uid
        train_metadata = deepcopy(args_dict)
        train_metadata["split_type"] = "train"
        test_metadata = deepcopy(args_dict)
        test_metadata["split_type"] = "test"
        metadata_uploads = [train_metadata, test_metadata]
        objects_to_upload = [augment_train_buffer, augment_test_buffer]
        object_uoload_names = [augment_train_filename, augment_test_filename]
        for obj, obj_nm, _metadata in zip(objects_to_upload, object_uoload_names, metadata_uploads):
            
            artifacts_list.append(ObjectToPersistData(upload_object=obj,
                                                        object_name=obj_nm,
                                                        metadata=_metadata,
                                                        bucket_name=args.augment_bucket_name
                                                        )
                                )
        
        logger.info("Starting to upload Augment artefacts to MinIO")
        upload_to_minio(minio_client=minio_client, 
                        objects_to_upload=artifacts_list,
                        bucket_name=None
                        )
        logger.info("Completed uploading Augment artefacts to MinIO")


if __name__ == "__main__":
    main()
    logger.info("Probability Augmentation completed successfully.")