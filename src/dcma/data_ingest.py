import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
import uuid
from minio import Minio
import io
import os
from datetime import datetime
import json
from dataclasses import dataclass
from typing import Union, List, Literal
from minio.error import S3Error
import inspect
import copy
import numpy as np

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)


@dataclass
class ObjectToPersistData:
        upload_object: io.BytesIO
        object_name: str
        metadata: dict
        bucket_name: Union[str,None] = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Data ingestion pipeline")
    parser.add_argument("--test_size", default=0.2, type=float, help="Proportion of the dataset to be used as test set")
    parser.add_argument("--random_state", default=2025, type=int, help="Random state for reproducing the split")
    parser.add_argument("--stratify_variable", required=False, default=None, help="Column name to use for stratification")
    parser.add_argument("--data_filepath", required=True, type=str, help="Path to the CSV data file")
    parser.add_argument("--shuffle", default=False, action="store_true", 
                        help="Whether to shuffle the dataset before splitting"
                        )
    parser.add_argument("--bucket_name", required=False, default="datasets", 
                        help="MinIO bucket to use (required if storing data in MinIO)"
                        )
    parser.add_argument("--access_key_env_name", default="MINIO_ACCESS_KEY", help="Env var name for MinIO access key")
    parser.add_argument("--access_secret_env_name", default="MINIO_SECRET_KEY", help="Env var name for MinIO secret key")
    parser.add_argument("--minio_server_url_env_name", default="MINIO_SERVER_URL", help="Env var name for MinIO server URL")
    parser.add_argument("--store_data_in_minio", default=False, action="store_true",
                        help="Flag to store data in MinIO instead of locally"
                        )
    parser.add_argument("--data_store_dir", default="metadata_store", 
                        help="Local directory to store data and metadata"
                        )
    parser.add_argument("--minio_endpoint_is_secured", action="store_true",
                        help="Whether the Minio endpoint url is secured"
                        )
    return parser.parse_args()

def get_minio_client(args)->Minio:
    ACCESS_KEY = os.getenv(key=args.access_key_env_name)
    ACCESS_SECRET = os.getenv(key=args.access_secret_env_name)
    MINIO_URL = os.getenv(key=args.minio_server_url_env_name)
    #minio_endpoint_is_secured = os.getenv(key=args.minio_endpoint_is_secured)
    
    
    env_vars = [args.access_key_env_name,
                args.access_secret_env_name,
                args.minio_server_url_env_name
                ]
    env_values = [ACCESS_KEY, ACCESS_SECRET, MINIO_URL]
    
    env_var_value_map = zip(env_vars, env_values)
    
    missing_env_vars = [var for var, value in env_var_value_map if not value]
    
    if missing_env_vars:
        logger.error(f"Missing MinIO environment variables: {missing_env_vars}")
        raise ValueError(f"MinIO credentials for {missing_env_vars} were not provided")
    
    client = Minio(endpoint=MINIO_URL, access_key=ACCESS_KEY, 
                   secret_key=ACCESS_SECRET,
                   secure=args.minio_endpoint_is_secured
                   )
    return client
    
def save_data_to_localdir(train_df, test_df, metadata,
                          train_file_name,
                          test_file_name, metadata_filepath
                          ):
    train_df.to_csv(train_file_name, index=False)
    test_df.to_csv(test_file_name, index=False)
    logger.info(msg=f"Saved data locally as train data: {train_file_name} and test data: {test_file_name}")
    
    with open(metadata_filepath, "w") as fp:
        json.dump(metadata, fp)
    logger.info(msg=f"Successfuly saved metadata locally at {metadata_filepath}")


def create_upload_object(train_df, test_df, metadata,
                        train_file_name, test_file_name,
                        ):
    train_buffer = io.BytesIO()
    test_buffer = io.BytesIO()
    train_df.to_csv(train_buffer, index=False)
    test_df.to_csv(test_buffer, index=False)
    train_buffer.seek(0)
    test_buffer.seek(0)
    
    reslist = []
    #metadata.copy()["split_type"] = "train"
    reslist.append(ObjectToPersistData(upload_object=train_buffer, 
                        object_name=train_file_name,
                        metadata=metadata
                        )
                   )
    #metadata.copy()["split_type"] = "test"
    reslist.append(ObjectToPersistData(upload_object=test_buffer, 
                                        object_name=test_file_name,
                                        metadata=metadata
                                        )
                   )
    return reslist
    
    
def upload_to_minio(minio_client, bucket_name, 
                    objects_to_upload: List[ObjectToPersistData],
                    **kwargs
                    ):
    
        
    for obj in objects_to_upload:
        bucket_name = obj.bucket_name if obj.bucket_name else bucket_name
        logger.info(f"Checking if bucket {bucket_name} exists")
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(msg=f"{bucket_name} does not exist hence was created")
            
        minio_client.put_object(bucket_name=bucket_name, 
                                object_name=obj.object_name,
                                data=obj.upload_object, 
                                length=obj.upload_object.getbuffer().nbytes,
                                metadata=obj.metadata
                                )
        logger.info(msg=f"{obj.object_name} successfully uploaded to {obj.bucket_name if obj.bucket_name else bucket_name}")



@dataclass
class MinioBucketRecords:
    object_name: str
    bucket_name: str
    metadata: dict
    version_id: str
    last_modified: datetime
    etag: str
    size: int
    owner_name: str
    owner_id: str
    is_dir: bool
    tags: str

def clean_metadata(metadata):
    """Remove the 'x-amz-meta-' prefix from custom metadata keys."""
    return {
        (key.replace("x-amz-meta-", "") if key.startswith("x-amz-meta-") else key): value 
        for key, value in metadata.items()
    }
    
    
def get_object_records(minio_client: Minio, bucket_name: str, object_name: str)-> MinioBucketRecords:
    logger.info(f"Start Getting records for {object_name} in {bucket_name} bucket")
    stat = minio_client.stat_object(bucket_name=bucket_name,
                                    object_name=object_name
                                    )
    obj_record = MinioBucketRecords(object_name=stat.object_name,
                                    bucket_name=stat.bucket_name,
                                    metadata=clean_metadata(stat.metadata),
                                    version_id=stat.version_id,
                                    last_modified=stat.last_modified,
                                    etag=stat.etag,
                                    size=stat.size,
                                    owner_name=stat.owner_name if hasattr(stat, "owner_name") else None,
                                    owner_id=stat.owner_id if hasattr(stat, "owner_id") else None,
                                    is_dir=stat.is_dir if hasattr(stat, "is_dir") else None,
                                    tags=stat.tags if hasattr(stat, "tags") else None
                                    )
    logger.info(f"Completed Getting records for {object_name} in {bucket_name} bucket")
    return obj_record   


def get_bucket_records(bucket_name, minio_client)-> List[MinioBucketRecords]:
    logger.info(f"Start Gettings records in {bucket_name} bucket")

    objs = minio_client.list_objects(bucket_name=bucket_name, 
                                     recursive=True, 
                                    include_user_meta=True, 
                                    include_version=True,
                                    fetch_owner=True
                                    )
    
    obj_records = []
    for obj in objs:
        obj_record = get_object_records(minio_client=minio_client,
                                            bucket_name=bucket_name,
                                            object_name=obj.object_name
                                            )
        obj_records.append(obj_record)
    logger.info(f"Completed Gettings records for all objects in {bucket_name} bucket")
    return obj_records
    
 
 
def download_from_minio(minio_client: Minio, bucket_name: str, object_name: str,
                        dytpe: Literal["csv", "npz"]
                        )-> pd.DataFrame: 
                        
    logger.info(f"Start downloading {object_name} from {bucket_name} bucket")
    try:
        obj_response = minio_client.get_object(bucket_name=bucket_name,
                                                object_name=object_name
                                                )
        data = obj_response.read()
        data_stream = io.BytesIO(data)
        data_stream.seek(0)
        if dytpe == "csv":
            df = pd.read_csv(data_stream)
        elif dytpe == "npz":
            df = np.load(data_stream)
        logger.info(f"Loaded {object_name} from {bucket_name} bucket")
        return df
    except S3Error as err:
        logger.error(f"Error downloading {object_name} from {bucket_name} bucket: {err}")
        raise ValueError(f"Error downloading {object_name} from {bucket_name} bucket: {err}")
        

def get_object_name(bucket_record, dataset_uid, split_type):
    if (bucket_record.metadata.get("uid") == dataset_uid) and (bucket_record.metadata["split_type"] == split_type):
        obj_name = bucket_record.object_name
        if not obj_name:
            logger.info(f"Retrieved {split_type} object name: {obj_name}")
    else:
        obj_name = None
    return obj_name


def get_positions(metadata: dict, variable_names: list, **kwargs):
    custom_msg = kwargs.pop("_log_msg", None)
    variables_not_found = [pred for pred in variable_names if pred not in metadata]
    if variables_not_found:
        logger.error(f"{custom_msg if custom_msg else ''} {variables_not_found} not found in metadata.")
        raise ValueError(f"{custom_msg if custom_msg else ''} {variables_not_found} not found in metadata.")
    var_positions = [metadata.get(var) for var in variable_names]
    logger.info(f"{custom_msg if custom_msg else ''} '{variable_names}' {'are' if len(variable_names) > 1 else 'is'} at position{f's {var_positions} (indices)' if len(var_positions)>1 else f' {var_positions} (index)'}")
    return var_positions




def get_variable_position_from_minio_metadata(bucket_record, variable: list, 
                                              split_type: str, 
                                              dataset_uid: str,
                                              **kwargs
                                              ):
    _log_msg = kwargs.pop("_log_msg", None)
    if (bucket_record.metadata.get("uid") == dataset_uid) and (bucket_record.metadata["split_type"] == split_type):
        var_position_metadata = bucket_record.metadata.get("column_position_map", {})
        var_position_metadata = json.loads(var_position_metadata) if isinstance(var_position_metadata, str) else var_position_metadata
        if isinstance(variable, str):
            variable = [variable]
        var_position = get_positions(metadata=var_position_metadata, variable_names=variable,
                                     _log_msg=_log_msg
                                     )
    else:
        var_position = None
    return var_position


def get_func_parameters(func):
    func_params = inspect.signature(func).parameters
    return func_params

def get_expected_params_for_func(func, **kwargs):
    func_params = get_func_parameters(func)
    expected_params = {param: value for param, value in kwargs.items() 
                       if param in func_params.keys()
                       }
    return expected_params
    
     
def main():
    args = parse_arguments()
    dataset_uuid = str(uuid.uuid4())
    if args.store_data_in_minio:
        minio_client = get_minio_client(args)
    
    try:
        data = pd.read_csv(args.data_filepath)
        logger.info(f"Read data successfully from {args.data_filepath}")
    except Exception as e:
        logger.error(f"Failed to read the data file:  {e}")
        raise ValueError(f"Failed to read data: {e}")
    
    try:
        train_df, test_df = train_test_split(data,
                                            test_size=float(args.test_size), 
                                            random_state=int(args.random_state),
                                            stratify=data[args.stratify_variable] if args.stratify_variable else None,
                                            shuffle=args.shuffle
                                            )
        logger.info(f"Completed data split with train shape: {train_df.shape} and test shape: {test_df.shape}")
    except Exception as e:
        logger.error(f"Failed Data Splitting with error: {e}")
        raise RuntimeError(f"Failed Data Splitting with error: {e}")
        
    creation_time = datetime.now()
    metadata = {"uid": dataset_uuid,
                "random_state": str(args.random_state),
                "test_size": str(args.test_size),
                "stratify_variable": str(args.stratify_variable),
                "shuffle": str(args.shuffle),
                "data_used_filepath": str(args.data_filepath),
                "creation_time": str(creation_time)
                }
    
    train_file_name = f"train_{dataset_uuid}.csv"
    test_file_name = f"test_{dataset_uuid}.csv"
    
    if not args.store_data_in_minio:
        train_file_name = os.path.join(args.data_store_dir, train_file_name)
        test_file_name = os.path.join(args.data_store_dir, test_file_name)
        metadata_filepath = os.path.join(args.data_store_dir, f"metadata_{dataset_uuid}.json")
        save_data_to_localdir(train_df=train_df, test_df=test_df,
                              metadata=metadata, train_file_name=train_file_name,
                              test_file_name=test_file_name,
                              metadata_filepath=metadata_filepath
                              )
    
    elif args.store_data_in_minio:
        objects_to_upload = create_upload_object(train_df=train_df, train_file_name=train_file_name,
                                                test_df=test_df, test_file_name=test_file_name,
                                                metadata=metadata
                                                )
        train_buffer = io.BytesIO()
        test_buffer = io.BytesIO()
        train_df.to_csv(train_buffer, index=False)
        test_df.to_csv(test_buffer, index=False)
        train_buffer.seek(0)
        test_buffer.seek(0)
        train_metdata = copy.deepcopy(metadata)
        train_metdata["split_type"] = "train"
        test_metdata = copy.deepcopy(metadata)
        test_metdata["split_type"] = "test"
        objects_to_upload = []
        for upload_obj, obj_name, obj_metadata in zip([train_buffer, test_buffer], 
                                                        [train_file_name, test_file_name], 
                                                        [train_metdata, test_metdata]
                                                        ):
            
            objects_to_upload.append(ObjectToPersistData(upload_object=upload_obj, 
                                                        object_name=obj_name,
                                                        metadata=obj_metadata
                                                        )
                                    )
        upload_to_minio(objects_to_upload=objects_to_upload,
                        metadata=metadata, bucket_name=args.bucket_name,
                        minio_client=minio_client
                        )
    
    
    
if __name__ == "__main__":
    main()
    
    
