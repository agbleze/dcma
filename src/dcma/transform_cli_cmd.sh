export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SERVER_URL="localhost:9000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

python transform_cli.py  --read_data_from_minio \
        --predictor_variables "category_id_encoded" "market_id_encoded" "customer_id_encoded" "publisher_encoded" "CPC" "['industry_embedding']" \
        --download_bucket_name "conversion-preprocess-storage" \
        --dataset_uid "35bffc4f-24eb-4ddf-ba22-0bc6fbcfc5e9" \
        --model_uri "s3://mlflow-artifacts/487d472b135d45bfa12f5850c7e5b358/artifacts/model" \
        --upload_output_to_minio --augment_bucket_name "augment-data-store"
        


     #--read_data_from_minio \