from transform import transform_data_with_conversion
import argparse
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Run Data transformation")
    parser.add_argument("--conversion_classifier_path", required=True)
    parser.add_argument("--read_data_from_minio",
                        type=bool
                        )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--predictors", required=True)
    
    
    args = parser.parse_args()
    
    train_cpa = transform_data_with_conversion(data=train_cpa, 
                                               variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )
    test_cpa = transform_data_with_conversion(data=test_cpa, variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )
    



