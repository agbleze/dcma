
# build docker image
docker build --file minio.Dockerfile -t my-minio .
docker run -p 9000:9000 -p 9001:9001 --name minio-instance -v /home/lin/codebase/dcma/data:/data my-minio