export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SERVER_URL="localhost:9000"

python train_cli.py --model_type "KNeighborsClassifier" --evaluation_metric "accuracy_score" \
        --scoring 'accuracy' "precision" "recall" "f1" --read_data_from_minio \
        --save_model_as "conversion_classifier.model" --download_bucket_name 