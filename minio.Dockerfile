FROM minio/minio

EXPOSE 9000 9001

VOLUME ["/data"]

ENV MINIO_ROOT_USER=minioadmin
ENV MINIO_ROOT_PASSWORD=minioadmin

CMD ["server", "/data", "--console-address", ":9001"]