from train import ModelTrainer
import argparse
from utils import read_minio_data
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run Model training")
    parser.add_argument("--model_registry", type=str, required=True)
    parser.add_argument("--read_data_from_minio",
                        type=bool
                        )
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--scoring", required=True, nargs="+")
    parser.add_argument("--evaluation_metric", required=True)
    parser.add_argument("--save_model_as", required=True)
    parser.add_argument("--cv", default=20)
    parser.add_argument("--model_result_metrics",
                        nargs="+"
                        )
    
    
    args = parser.parse_args()
    








