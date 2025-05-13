from minio import Minio
from minio.error import S3Error

# Initialize the client with your MinIO server details
client = Minio(
    "localhost:9000",  # or your MinIO server address
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # set to True if using HTTPS
)

# List all buckets
try:
    buckets = client.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)
except S3Error as err:
    print("Error listing buckets:", err)

# Create a new bucket if it does not already exist
bucket_name = "my-bucket"
try:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created successfully")
    else:
        print(f"Bucket '{bucket_name}' already exists")
except S3Error as err:
    print("Error creating bucket:", err)
