
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SERVER_URL="localhost:9000"
python data_ingest.py --test_size 0.2 --random_state 2025 --data_filepath "/home/lin/codebase/dcma/src/dcma/Data Science Challenge - ds_challenge_data.csv" \
        --shuffle --bucket_name "rawdata" --store_data_in_minio

