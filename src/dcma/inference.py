from preprocess import (preprocess_input_for_conversion_classifier,
                        preprocess_input_for_cpa_model
                        )

from utils import get_model
import pandas as pd
import numpy as np
from typing import List

def predict_conversion_proba(predictors, classifier_model):
    proba = classifier_model.predict_proba(predictors)[:, 1]
    proba
    

def predict_cpa(convert_model_path: str, cpa_model_path: str,
                customer_id: str, category_id: str,
                industry: str, publisher_id: str,
                market_id: str,
                customer_id_freq_record: pd.DataFrame,
                category_id_freq_record: pd.DataFrame,
                market_id_freq_record: pd.DataFrame, 
                publisher_freq_record: pd.DataFrame,
                cpc_per_market_id_publisher_records: pd.DataFrame,
                values_embeddings_map={},
                )->List[float]:
    convert_model = get_model(convert_model_path)
    cpa_model = get_model(cpa_model_path)
    preprocessed_input = preprocess_input_for_conversion_classifier(customer_id_freq=customer_id_freq_record,
                                                                    customer_id=customer_id,
                                                                    category_id=category_id,
                                                                    category_id_freq=category_id_freq_record,
                                                                    industry=industry,
                                                                    values_embeddings_map=values_embeddings_map,
                                                                    market_id_freq=market_id_freq_record, 
                                                                    market_id=market_id,
                                                                    publisher_freq=publisher_freq_record,
                                                                    publisher_id=publisher_id,
                                                                    cpc_per_market_id_publisher_records=cpc_per_market_id_publisher_records
                                                                    )
    
    preprocessed_data = preprocessed_input["preprocessed_data"]
    preprocessed_predictors = preprocessed_input["preprocessed_predictors"]
    
    proba = convert_model.predict_proba(preprocessed_predictors)[:, 1]
    preprocessed_cpa_input = preprocess_input_for_cpa_model(proba_to_convert=proba, 
                                                            input_data=preprocessed_data
                                                            )
    
    
    cpa_pred = cpa_model.predict(preprocessed_cpa_input)
    return np.expm1(cpa_pred)
