export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SERVER_URL="localhost:9000"

python preprocess_cli.py --categorical_features 'category_id' 'market_id' 'customer_id' 'publisher' \
        --numeric_features "CPC" --features_to_embed "industry" --target_variable "convert" \
        --encoder_type "frequency_encoding" --read_data_from_minio --upload_output_to_minio \
        --dataset_uid "8c5223ef-5e2e-4930-87fc-590749507d0f" --categorical_target "convert" \
        --stats_to_compute "count" --local_metadata_store "metadata_store" \
        --local_preprocess_store "preprocess_store" --local_feature_encoder_store "feature_encoder_store" \
        --download_bucket_name "rawdata" --feature_encoder_bucket_name "feature-encoder-store" \
        --preprocess_bucket_name "conversion-preprocess-storage" \
        --save_dir "metadata_store"

