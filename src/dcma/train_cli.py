from train import ModelTrainer
import argparse
from utils import read_minio_data
import pandas as pd
from data_ingest import (get_bucket_records, download_from_minio,
                         get_minio_client,
                         get_expected_params_for_func
                         )
import logging
import numpy as np
import json
import os


logging.basicConfig(level=logging.INFO,
                             format="%(asctime)s - %(levelname)s - %(message)s"
                            )

logger = logging.getLogger(__name__)

def parse_argumments():
    parser = argparse.ArgumentParser(description="Run Model training")
    parser.add_argument("--model_registry", type=str, default="mlflow",
                        help="Model registry to use. Currently only mlflow is supported"
                        )
    parser.add_argument("--read_data_from_minio", action="store_true",
                        help="Whether to read data from MinIO"
                        )
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--scoring", required=True, nargs="+")
    parser.add_argument("--evaluation_metric", required=True)
    parser.add_argument("--save_model_as", required=True)
    parser.add_argument("--cv", default=20)
    parser.add_argument("--model_result_metrics",
                        nargs="+"
                        )
    parser.add_argument("--download_bucket_name", type=str)
    parser.add_argument("--dataset_uid", type=str)
    parser.add_argument("--upload_output_to_minio", action="store_true")
    parser.add_argument("--access_key_env_name", default="MINIO_ACCESS_KEY", help="Env var name for MinIO access key")
    parser.add_argument("--access_secret_env_name", default="MINIO_SECRET_KEY", help="Env var name for MinIO secret key")
    parser.add_argument("--minio_server_url_env_name", default="MINIO_SERVER_URL", help="Env var name for MinIO server URL")
    parser.add_argument("--minio_endpoint_is_secured", action="store_true",
                        help="Whether the Minio endpoint url is secured"
                        )
    parser.add_argument("--local_train_data_path", type=str)
    parser.add_argument("--local_test_data_path", type=str)
    parser.add_argument("--train_preprocessed_metadata_filepath", type=str)
    parser.add_argument("--include_sample_weight", action="store_true",
                        help="Whether to include sample weight in the training"
                        )
    
    
    return parser.parse_args()
    

def main():
    args = parse_argumments()
    logger.info(f"Arguments: {args}")
    
    if args.read_data_from_minio or args.upload_output_to_minio:
        minio_client = get_minio_client(args=args)
    if args.read_data_from_minio:
        
        bucket_records = get_bucket_records(bucket_name=args.download_bucket_name,
                                            minio_client=minio_client
                                            )
        print(bucket_records[0].metadata["uid"])
        train_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
                            and (bc.metadata["split_type"] == "train")
                        ]
        logger.info(f"Retrieved train object name: {train_obj_name}")
        test_obj_name = [bc.object_name for bc in bucket_records if (bc.metadata.get("uid") == args.dataset_uid) 
                            and (bc.metadata["split_type"] == "test")
                        ]
        logger.info(f"Retrieved test object name: {test_obj_name}")
        
        if args.include_sample_weight:
            train_class_weight = [bc.metadata.get("class_weight") for bc in bucket_records if bc.object_name == train_obj_name[0]
                                ][0]
            
        
        train_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=train_obj_name[0],
                                        dytpe="npz"
                                        )
        test_data = download_from_minio(minio_client=minio_client, 
                                        bucket_name=args.download_bucket_name, 
                                        object_name=test_obj_name[0],
                                        dytpe="npz"
                                        )
    elif not args.read_data_from_minio:
        train_data = np.load(args.local_train_data_path)
        test_data = np.load(args.local_test_data_path)
        
        with open(args.train_preprocessed_metadata_filepath, "rb") as f:
            train_preprocessed_metadata = json.load(f)
        train_class_weight = train_preprocessed_metadata.get("class_weight")
            
        
    train_npz = train_data.get("preprocessed_data")
    training_target = train_npz[:, 0]
    training_predictors = train_npz[:, 1:]
    
    test_npz = test_data.get("preprocessed_data")
    testing_target = test_npz[:, 0]
    testing_predictors = test_npz[:, 1:]
    
    if args.include_sample_weight:
        train_sample_weight = np.array([train_class_weight[int(label)] for label in training_target])
    else:
        train_sample_weight = None
    print("############# running model  ##########")
    trainer = ModelTrainer(training_predictors=training_predictors,
                            training_target=training_target,
                            testing_predictors=testing_predictors,
                            testing_target=testing_target,
                            sample_weight=train_sample_weight,
                            model_type=args.model_type,
                            #model_registry=args.model_registry,
                            )
    trainer.run_model_training_pipeline(cv=20, 
                                        scoring=args.scoring, #['accuracy', "precision", "recall", "f1"],
                                        model_result_metrics=args.model_result_metrics,
                                        
                                        # ['test_accuracy',  'train_accuracy',
                                        #                       'test_precision', 'train_precision',
                                        #                       'test_recall', 'train_recall',
                                        #                       'test_f1', 'train_f1'
                                        #                       ],
                                        evaluation_metric=args.evaluation_metric, #"accuracy_score",
                                        save_model_as=args.save_model_as, #"conversion_proba.model",
                                        save_dir="model_store"
                                        )
    

if __name__ == "__main__":
    main()






