import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
import uuid
from minio import Minio
import io


def main():
    parser = argparse.ArgumentParser(description="Run Data ingestion pipeline")
    parser.add_argument("--test_size", default=0.2)
    parser.add_argument("--random_state", default=2025)
    parser.add_argument("--stratify_variable", required=False, default=None)
    parser.add_argument("--data_filepath", required=True)
    parser.add_argument("--shuffle", default=True, required=False)
    
    
    args = parser.parse_args()

    data = pd.read_csv(args.data_filepath)
    train, test = train_test_split(data,
                                    test_size=args.test_size, 
                                    random_state=args.random_state,
                                    stratify=args.stratify_variable,
                                    shuffle=args.shuffle
                                    )