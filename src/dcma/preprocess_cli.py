import argparse
from utils import read_minio_data
import pandas as pd
from preprocess import (compute_cpa, compute_cpc,
                        PreprocessPipeline
                        )
from transform import (create_binary_conversion_variable,
                       transform_data_with_conversion
                       )
import uuid
import io
import numpy as np
from data_ingest import (get_minio_client, upload_to_minio, ObjectToPersistData,
                         get_bucket_records, download_from_minio,
                         get_expected_params_for_func
                         )
import copy
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run preprocess pipeline")
    parser.add_argument("--categorical_features", type=str,
                        nargs="+",
                        required=True
                        )
    parser.add_argument("--numeric_features", 
                        type=str,
                        nargs="+",
                        required=True
                        )
    parser.add_argument("--target_variable", type=str, required=True)
    parser.add_argument("--features_to_embed", type=str, 
                        nargs="+",
                        required=True
                        )
    parser.add_argument("--categorical_target", type=str,
                        #nargs="+",
                        required=False
                        )
    parser.add_argument("--encoder_type",
                        type=str,
                        default="frequency_encoding",
                        required=True
                        )
    parser.add_argument("--stats_to_compute",
                        type=str,
                        default="count"
                        )
    parser.add_argument("--save_dir",
                        type=str,
                        default="metadata_store",
                        )
    parser.add_argument("--local_metadata_store", type=str, default="metadata_store")
    parser.add_argument("--local_preprocess_store", type=str, default="preprocess_store")
    parser.add_argument("--local_feature_encoder_store", type=str, default="feature_encoder_store")
    parser.add_argument("--train_data_path",
                        type=str
                        )
    parser.add_argument("--test_data_path",
                        type=str
                        )
    parser.add_argument("--read_data_from_minio",
                        action="store_true",
                        help="Flag to read data from MinIO"
                        )
    #parser.add_argument("--save_bucket", type=str)
    parser.add_argument("--download_bucket_name", type=str)
    parser.add_argument("--dataset_uid", type=str)
    parser.add_argument("--upload_output_to_minio", action="store_true")
    parser.add_argument("--feature_encoder_bucket_name", type=str)
    parser.add_argument("--preprocess_bucket_name", type=str)
    parser.add_argument("--access_key_env_name", default="MINIO_ACCESS_KEY", help="Env var name for MinIO access key")
    parser.add_argument("--access_secret_env_name", default="MINIO_SECRET_KEY", help="Env var name for MinIO secret key")
    parser.add_argument("--minio_server_url_env_name", default="MINIO_SERVER_URL", help="Env var name for MinIO server URL")
    parser.add_argument("--minio_endpoint_is_secured", action="store_true",
                        help="Whether the Minio endpoint url is secured"
                        )
    
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    print(args.categorical_features)
    print(args.numeric_features)
    preprocess_dataset_uuid = uuid.uuid4()
    encoder_uuid = uuid.uuid4()
    if args.read_data_from_minio or args.upload_output_to_minio:
        minio_client = get_minio_client(args=args)
        
    if args.read_data_from_minio:
        
        bucket_records = get_bucket_records(bucket_name=args.download_bucket_name,
                                            minio_client=minio_client
                                            )
        print(bucket_records[0].metadata["uid"])
        train_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
                            and (bc.metadata["split_type"] == "train")
                        ]
        logger.info(f"Retrieved train object name: {train_obj_name}")
        test_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
                            and (bc.metadata["split_type"] == "test")
                        ]
        logger.info(f"Retrieved test object name: {test_obj_name}")
        
        train_df = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=train_obj_name[0],
                                        dytpe="csv"
                                        )
        test_df = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=test_obj_name[0],
                                        dytpe="csv"
                                        )
    else:
        train_df = pd.read_csv(args.data_path)
        test_df = pd.read_csv(args.test_data_path)
    
    logger.info("Computing CPA and CPC")
    train_df = compute_cpa(data=train_df)
    train_df = create_binary_conversion_variable(train_df)
    train_df = compute_cpc(data=train_df)
    
    test_df = compute_cpa(data=test_df)
    test_df = create_binary_conversion_variable(test_df)
    test_df = compute_cpc(data=test_df)
    logger.info("Completed CPA and CPC computed")
    
    
    # categorical_features = ['category_id', 'market_id',
    #                         'customer_id', 'publisher',
    #                         ]
    # numeric_features = ["CPC"]
    # features_to_embed = "industry"
    logger.info("Initialize PreprocessPipeline")
    preprocess_pipeline = PreprocessPipeline(data=train_df, 
                                             categorical_features=args.categorical_features,
                                            numeric_features=args.numeric_features,
                                            target_variable=args.target_variable,
                                            features_to_embed=args.features_to_embed
                                            )
    logger.info("Started Running train data PreprocessPipeline")
    preprocessed_train_datastore = preprocess_pipeline.run_preprocess_pipeline(categorical_target=args.categorical_target, #"convert",
                                                                               encoder_type=args.encoder_type, #"frequency_encoding",
                                                                               save_dir=args.save_dir, 
                                                                               stats_to_compute=args.stats_to_compute
                                                                              )
    logger.info("Completed Running train data PreprocessPipeline")
    
    #sample_weight = preprocess_pipeline.sample_weight
    class_weight = preprocess_pipeline.class_weight_dict
    train_preprocessed_data = preprocessed_train_datastore.full_data
    train_preprocessed_columns_inorder = preprocessed_train_datastore.full_data_columns_in_order
    logger.info(f"train_preprocessed_data shape: {train_preprocessed_data.shape}")
    logger.info(f"train_preprocessed_columns_inorder: {train_preprocessed_columns_inorder}")
    logger.info(f"class weight: {class_weight}")
    
    preprocessed_train_filename = f"train_{preprocess_dataset_uuid}.npz"
    train_preprocessed_metadata = {"split_type": "train",
                                    "categorical_features": args.categorical_features,
                                    "numeric_features": args.numeric_features,
                                    "target_variable": args.target_variable,
                                    "features_to_embed": args.features_to_embed,
                                    "encoder_uid": str(encoder_uuid),
                                    "parent_uid": args.dataset_uid,
                                    "uid": preprocess_dataset_uuid,
                                    "column_order": train_preprocessed_columns_inorder,
                                    #"file_name": preprocessed_train_filename,
                                    "class_weight": class_weight, #str(sample_weight.tolist())
                                    "class_weight_based_on": args.categorical_target
                                    }
    train_preprocessed_metadata_file = f"train_preprocessed_metadata_{preprocess_dataset_uuid}.json"
    
    ### prepare encode features for upload
    artifacts_list = []
    feat_store = preprocess_pipeline.feat_encoder_store
    encode_features = [(f"{catvar}_{preprocess_dataset_uuid}.csv", getattr(feat_store, catvar))  
                       for catvar in 
                       preprocess_pipeline.categorical_features
                       ]
    
    feature_encoded_metadata = {"categorical_target": args.categorical_target, #"convert",
                                "encoder_type": args.encoder_type, #"frequency_encoding",
                                "stats_to_compute": args.stats_to_compute, #"count",
                                "uid": str(encoder_uuid),
                                "parent_uid": str(args.dataset_uid)
                                }
    feature_encoded_metadata_filepath = os.path.join(f"{args.local_feature_encoder_store}", 
                                                     f"feature_encoded_metadata_{encoder_uuid}.json"
                                                     )
    
    logger.info("Start Preprocessing test data")
    preprocessed_test_datastore = preprocess_pipeline.run_preprocess_pipeline(categorical_target="convert",
                                                                                encoder_type="frequency_encoding",
                                                                                save_dir="metadata_store",
                                                                                stats_to_compute="count",
                                                                                data=test_df
                                                                                )
    test_preprocessed_data = preprocessed_test_datastore.full_data
    test_preprocessed_columns_inorder = preprocessed_test_datastore.full_data_columns_in_order
    logger.info(f"test_preprocessed_data shape: {test_preprocessed_data.shape}")
    logger.info(f"test_preprocessed_columns_inorder: {test_preprocessed_columns_inorder}")
    
    test_preprocessed_metadata = copy.deepcopy(train_preprocessed_metadata)
    preprocessed_test_filename = f"test_{preprocess_dataset_uuid}.npz"
    test_preprocessed_metadata["split_type"] = "test"
    
    test_preprocessed_metadata["column_order"] = test_preprocessed_columns_inorder
    test_preprocessed_metadata_file = f"test_preprocessed_metadata_{preprocess_dataset_uuid}.json"
    logger.info("Completed Preprocessing test data")
    
    if not args.upload_output_to_minio:
        logger.info("Start saving preprocessed data locally")
        np.savez_compressed(file=os.path.join(f"{args.local_preprocess_store}", {preprocessed_train_filename}), 
                            preprocessed_data=train_preprocessed_data
                            )
        np.savez_compressed(file=os.path.join(f"{args.local_preprocess_store}", preprocessed_test_filename), 
                            preprocessed_data=test_preprocessed_data
                            )
        logger.info("Completed saving preprocessed data locally")
        
        logger.info("Start saving preprocess metadata locally")
        train_preprocessed_metadata_filepath = os.path.join(f"{args.local_metadata_store}", 
                                                            train_preprocessed_metadata_file
                                                            )
        test_preprocessed_metadata_filepath = os.path.join(f"{args.local_metadata_store}",
                                                            test_preprocessed_metadata_file
                                                            )
        with open(train_preprocessed_metadata_filepath, "w") as fp:
            json.dump(train_preprocessed_metadata, fp)
        
        with open(test_preprocessed_metadata_filepath, "w") as fp:
            json.dump(test_preprocessed_metadata, fp)
        logger.info("Completed saving preprocess metadata locally")
        
        
        logger.info("Start saving feature encoded data and metadata locally")
        for feat_filename, feat in encode_features:
            feat.to_csv(os.path.join(f"{args.local_feature_encoder_store}", feat_filename), index=False)
            
        with open(feature_encoded_metadata_filepath, "w") as fp:
            json.dump(feature_encoded_metadata, fp)
        logger.info("Completed saving feature encoded data and metadata locally")
        
        
    if args.upload_output_to_minio:
        logger.info("Start creating feature encode objects for upload")
        
        for feat_filename,  feat in encode_features:
            feat_buffer = io.BytesIO()
            feat.to_csv(feat_buffer, index=False)
            feat_buffer.seek(0)
            artifacts_list.append(ObjectToPersistData(upload_object=feat_buffer,
                                                        object_name=feat_filename,
                                                        metadata=feature_encoded_metadata,
                                                        bucket_name=args.feature_encoder_bucket_name
                                                        )
                                        )
        logger.info("Completed creating feature encode objects for upload")
        
        logger.info("Start creating preprocessed data objects for upload")
        preprocessed_train_buffer = io.BytesIO()
        np.savez_compressed(file=preprocessed_train_buffer, 
                            preprocessed_data=train_preprocessed_data
                            )
        preprocessed_train_buffer.seek(0)
        
        artifacts_list.append(ObjectToPersistData(upload_object=preprocessed_train_buffer,
                                                    object_name=preprocessed_train_filename,
                                                    metadata=train_preprocessed_metadata,
                                                    bucket_name=args.preprocess_bucket_name
                                                    )
                            )
        
        preprocessed_test_buffer = io.BytesIO()
        np.savez_compressed(file=preprocessed_test_buffer, 
                            preprocessed_data=test_preprocessed_data
                            )
        preprocessed_test_buffer.seek(0)
        
        artifacts_list.append(ObjectToPersistData(upload_object=preprocessed_test_buffer,
                                                object_name=preprocessed_test_filename,
                                                metadata=test_preprocessed_metadata,
                                                bucket_name=args.preprocess_bucket_name
                                                )
                            )
        logger.info("Completed creating preprocessed data objects for upload")
        
        logger.info("Start Uploading preprocessed artefacts to MinIO")
        upload_to_minio(minio_client=minio_client, objects_to_upload=artifacts_list,
                        bucket_name=None
                        )
        logger.info("Completed uploading Preprocess artefacts to MinIO")
        


if __name__ == "__main__":
    main()