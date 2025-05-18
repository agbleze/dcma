import argparse
from utils import read_minio_data
import pandas as pd
from preprocess import (compute_cpa, compute_cpc,
                        PreprocessPipeline
                        )
from transform import (create_binary_conversion_variable,
                       transform_data_with_conversion
                       )

def main():
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
    
    args = parser.parse_args()
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
    preprocessed_train_target_convert = preprocessed_train_datastore.target
    preprocessed_train_predictors_convert = preprocessed_train_datastore.predictors
    preprocessed_train_predictor_colnames_inorder = preprocessed_train_datastore.predictor_colnames_inorder
    sample_weight = preprocess_pipeline.sample_weight
    X_train_encoded_embedded_data = preprocess_pipeline.encoded_embedded_data
    train_preprocessed_data = preprocessed_train_datastore.full_data
    train_preprocessed_columns_inorder = preprocessed_train_datastore.full_data_columns_in_order
    
    ### use the encoders created for train dataset to encode test dataset 
    # and prepare it for model evaluation. This prevents data leakage and 
    # ensures model evaluation reflect model performance in terms of the encoding 
    # used during training
    X_test_convert_encoded = preprocess_pipeline.encode_features(data=test_df,
                                                                 stats_to_compute="count"
                                                                 ) 
    X_test_convert_encoded_embed = preprocess_pipeline.transform_columns_to_embed(data=X_test_convert_encoded)
    
    return X_train_encoded_embedded_data, preprocessed_train_target_convert, X_test_convert_encoded_embed
    
    
    preprocess_test_datastore = preprocess_pipeline.prepare_modelling_data(predictors=preprocess_pipeline.predictors,
                                                                            embedding_colname=preprocess_pipeline.embedding_colname,
                                                                            data=X_test_convert_encoded_embed,
                                                                            target=preprocess_pipeline.target_variable
                                                                            )
    
    preprocessed_test_predictors_convert = preprocess_test_datastore.predictors
    preprocessed_test_target_convert = preprocess_test_datastore.target
    preprocessed_test_predictor_colnames_inorder = preprocess_test_datastore.predictor_colnames_inorder
    

