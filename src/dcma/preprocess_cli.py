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
from data_ingest import get_minio_client, upload_to_minio, ObjectToPersistData
import copy


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
                        nargs="+",
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
                        default="save_dir"
                        )
    parser.add_argument("--data_path",
                        type=str
                        )
    parser.add_argument("--test_data_path",
                        type=str
                        )
    parser.add_argument("--read_data_from_minio",
                        type=bool
                        )
    parser.add_argument("--save_bucket", type=str)
    parser.add_argument("--dataset_uid")
    parser.add_argument("--feature_encoder_bucket_name")
    parser.add_argument("--preprocess_bucket_name")
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    
    preprocess_dataset_uuid = uuid.uuid4()
    
    if args.read_data_from_minio:
        train_df = read_minio_data(data_path=args.data_path)
        test_df = read_minio_data(data_path=args.test_data_path)
    else:
        train_df = pd.read_csv(args.data_path)
        test_df = pd.read_csv(args.test_data_path)
        
    train_df = compute_cpa(data=train_df)
    train_df = create_binary_conversion_variable(train_df)
    train_df = compute_cpc(data=train_df)
    
    test_df = compute_cpa(data=test_df)
    test_df = create_binary_conversion_variable(test_df)
    test_df = compute_cpc(data=test_df)
    
    
    categorical_features = ['category_id', 'market_id',
                            'customer_id', 'publisher',
                            ]
    numeric_features = ["CPC"]
    features_to_embed = "industry"
    preprocess_pipeline = PreprocessPipeline(data=train_df, 
                                             categorical_features=categorical_features,
                                            numeric_features=numeric_features,
                                            target_variable="convert",
                                            features_to_embed=features_to_embed
                                            )
    print("############  preprocessing  ###########")
    preprocessed_train_datastore = preprocess_pipeline.run_preprocess_pipeline(categorical_target="convert",
                                                                               encoder_type="frequency_encoding",
                                                                               save_dir="metadata_store",
                                                                               stats_to_compute="count"
                                                                              )
    # preprocessed_train_target_convert = preprocessed_train_datastore.target
    # preprocessed_train_predictors_convert = preprocessed_train_datastore.predictors
    # preprocessed_train_predictor_colnames_inorder = preprocessed_train_datastore.predictor_colnames_inorder
    sample_weight = preprocess_pipeline.sample_weight
    encoder_uuid = uuid.uuid4()
    feature_encoded_metadata = {"categorical_target": "convert",
                                "encoder_type": "frequency_encoding",
                                "stats_to_compute": "count",
                                "uid": str(encoder_uuid),
                                "parent_uid": str(args.dataset_uuid)
                                }
    ### prepare encode features for upload
    artifacts_list = []
    feat_store = preprocess_pipeline.feat_encoder_store
    encode_features = [(f"{catvar}_{preprocess_dataset_uuid}.csv", getattr(feat_store, catvar))  
                       for catvar in 
                       preprocess_pipeline.categorical_features
                       ]
    feat_buffer = io.BytesIO()
    for feat_filename,  feat in encode_features:
        feat.to_csv(feat_buffer, index=False)
        feat_buffer.seek(0)
        artifacts_list.append(ObjectToPersistData(upload_object=feat_buffer,
                                                    object_name=feat_filename,
                                                    metadata=feature_encoded_metadata,
                                                    bucket_name=args.feature_encoder_bucket_name
                                                    )
                                      )
    
    
    #X_train_encoded_embedded_data = preprocess_pipeline.encoded_embedded_data
    train_preprocessed_data = preprocessed_train_datastore.full_data
    train_preprocessed_columns_inorder = preprocessed_train_datastore.full_data_columns_in_order
    
    preprocessed_train_buffer = io.BytesIO()
    np.savez_compressed(file=preprocessed_train_buffer, 
                        preprocessed_data=train_preprocessed_data
                        )
    preprocessed_train_buffer.seek(0)
    preprocessed_train_filename = f"train_{preprocess_dataset_uuid}.npz"
    
    train_preprocessed_metadata = {"split_type": "train",
                                    "categorical_features": categorical_features,
                                    "numeric_features": numeric_features,
                                    "target_variable": "convert",
                                    "features_to_embed": features_to_embed,
                                    "encoder_uid": str(encoder_uuid),
                                    "parent_uid": args.dataset_uuid,
                                    "uid": preprocess_dataset_uuid,
                                    "column_order": str(train_preprocessed_columns_inorder),
                                    "file_name": preprocessed_train_filename,
                                    "sample_weight": str(sample_weight.tolist())
                                    }
    artifacts_list.append(ObjectToPersistData(upload_object=preprocessed_train_buffer,
                                                object_name=preprocessed_train_filename,
                                                metadata=train_preprocessed_metadata,
                                                bucket_name=args.preprocess_bucket_name
                                                )
                          )
    
    ### use the encoders created for train dataset to encode test dataset 
    # and prepare it for model evaluation. This prevents data leakage and 
    # ensures model evaluation reflect model performance in terms of the encoding 
    # used during training
    # X_test_convert_encoded = preprocess_pipeline.encode_features(data=test_df,
    #                                                              stats_to_compute="count"
    #                                                              ) 
    # X_test_convert_encoded_embed = preprocess_pipeline.transform_columns_to_embed(data=X_test_convert_encoded)
    
    
    preprocessed_test_datastore = preprocess_pipeline.run_preprocess_pipeline(categorical_target="convert",
                                                                                encoder_type="frequency_encoding",
                                                                                save_dir="metadata_store",
                                                                                stats_to_compute="count",
                                                                                data=test_df
                                                                                )
    test_preprocessed_data = preprocessed_test_datastore.full_data
    test_preprocessed_columns_inorder = preprocessed_test_datastore.full_data_columns_in_order
    
    preprocessed_test_buffer = io.BytesIO()
    np.savez_compressed(file=preprocessed_test_buffer, 
                        preprocessed_data=test_preprocessed_data
                        )
    preprocessed_test_buffer.seek(0)
    preprocessed_test_filename = f"test_{preprocess_dataset_uuid}.npz"
    
    
    test_preprocessed_metadata = copy.deepcopy(train_preprocessed_metadata)
    test_preprocessed_metadata["split_type"] = "test"
    test_preprocessed_metadata["file_name"] = preprocessed_test_filename
    test_preprocessed_metadata["column_order"] = test_preprocessed_columns_inorder
    
    artifacts_list.append(ObjectToPersistData(upload_object=preprocessed_test_buffer,
                                              object_name=preprocessed_test_filename,
                                              metadata=test_preprocessed_metadata,
                                              bucket_name=args.preprocess_bucket_name
                                              )
                          )
    minio_client = get_minio_client(args=args)
    upload_to_minio(minio_client=minio_client, objects_to_upload=artifacts_list,
                    bucket_name=None
                    )
    
