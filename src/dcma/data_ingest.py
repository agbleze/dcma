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
from typing import Union, List

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
    return parser.parse_args()

def get_minio_client(args)->Minio:
    ACCESS_KEY = os.getenv(key=args.access_key_env_name)
    ACCESS_SECRET = os.getenv(key=args.access_secret_env_name)
    MINIO_URL = os.getenv(key=args.minio_server_url_env_name)
    
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
                   secure=False
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
    reslist.append(ObjectToPersistData(upload_object=train_buffer, 
                        object_name=train_file_name,
                        metadata=metadata
                        )
                   )
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
    # train_buffer = io.BytesIO()
    # test_buffer = io.BytesIO()
    # train_df.to_csv(train_buffer, index=False)
    # test_df.to_csv(test_buffer, index=False)
    # train_buffer.seek(0)
    # test_buffer.seek(0)
    
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(msg=f"{bucket_name} does not exist hence was created")
        
    for obj in objects_to_upload:
        minio_client.put_object(bucket_name=obj.bucket_name if obj.bucket_name else bucket_name, 
                                object_name=obj.object_name,
                                data=obj.upload_object, 
                                length=obj.upload_object.getbuffer().nbytes,
                                metadata=obj.metadata
                                )
        logger.info(msg=f"{obj.object_name} successfully uploaded to {obj.bucket_name if obj.bucket_name else bucket_name}")
        
        
    # minio_client.put_object(bucket_name=bucket_name,
    #                         object_name=test_file_name,
    #                         data=test_buffer,
    #                         length=test_buffer.getbuffer().nbytes,
    #                         metadata=metadata
    #                         )
    # logger.info(msg=f"{test_file_name} successfully uploaded to {bucket_name}")
    
    
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
        upload_to_minio(objects_to_upload=objects_to_upload,
                        metadata=metadata, bucket_name=args.bucket_name,
                        minio_client=minio_client
                        )
    
if __name__ == "__main__":
    main()
    
    
