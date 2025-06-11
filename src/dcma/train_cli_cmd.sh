export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SERVER_URL="localhost:9000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

python train_cli.py --model_type "KNeighborsClassifier" --evaluation_metric "accuracy_score" \
        --scoring 'accuracy' "precision" "recall" "f1" --read_data_from_minio \
        --save_model_as "conversion_classifier.model" --target_variable "convert" \
        --predictor_variables "category_id_encoded" "market_id_encoded" "customer_id_encoded" "publisher_encoded" "CPC" "['industry_embedding']" \
        --download_bucket_name "conversion-preprocess-storage" \
        --mlflow_tracking_uri "http://localhost:7000" --dataset_uid "35bffc4f-24eb-4ddf-ba22-0bc6fbcfc5e9" \
        --model_result_metrics "test_accuracy" "train_accuracy" "test_precision" "train_precision" 'test_recall' 'train_recall'\
                                'test_f1' 'train_f1'
        


