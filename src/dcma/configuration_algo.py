import pandas as pd
from typing import List, Dict, Union
from utils import MarketPublisherConfig
from preprocess import (preprocess_input_for_conversion_classifier,
                        preprocess_input_for_cpa_model
                        )


###### CONCEPT  -- Partially implemented  ########

"""
Decision-Making Configuration Algorithm - DMCA

1. Generate configuration
-- Explore vs Exploit
 -- Exploit exisitng configuration space
Fetch existing market_id and publisher configurations decide on number -- 100 most used configuration

-- Explore
Generate new configuration combination f market_id and publisher 

-- Hybrid
Combine explore and exploit configration results
For each input + configuration result
2. preprocess input data and configuration for conversion probability model
3. Predict conversion probability
4. Augment preprocessed input with conversion probability
5. Use augmented to predict CPA and store result
6. Choose configuration with lowest CPA as recommendation

"""

def get_top_k_configurations(market_id_publisher_config_record: pd.DataFrame, 
                              k: int = 5, 
                              filter_colname="campaign_id"
                              ) -> pd.DataFrame:
    """Gets the top k configurations from the market_id_publisher_config_data

    Args:
        market_id_publisher_config_data (pd.DataFrame): Data with configurations
        k (int, optional): Number of top configurations to get. Defaults to 5.

    Returns:
        pd.DataFrame: Data with top k configurations
    """
    if filter_colname not in market_id_publisher_config_record.columns:
        raise ValueError(f"Data must contain {filter_colname} column to compute top configurations.")
    return market_id_publisher_config_record.nlargest(k, columns=filter_colname)


def recommend_configuration(configurations: List[MarketPublisherConfig]) -> MarketPublisherConfig:
    """Recommends the best configuration based on the lowest CPA

    Args:
        configurations (list): List of configurations

    Returns:
        MarketPublisherConfig: Best configuration
    """
    if not configurations:
        raise ValueError("No configurations available to recommend.")
      
        
    config_cpa = [config.cpc for config in configurations]
    lowest_cpa_index = config_cpa.index(min(config_cpa))
    best_config = configurations[lowest_cpa_index]
    print(f"Recommended Configuration: market_id - {best_config.market_id}, publisher - {best_config.publisher}")
    return best_config  



def simulate_configuration(cpc_per_market_id_publisher_records: pd.DataFrame,
                            market_id_freq: pd.DataFrame,
                            publisher_freq: pd.DataFrame,
                            customer_id_freq: pd.DataFrame,
                            category_id_freq: pd.DataFrame,
                            category_id: str,
                            customer_id: str,
                            industry: str,
                            conversion_classifier: object,
                            configuration_search_space: pd.DataFrame,
                            cpa_model: object,
                            values_embeddings_map: Dict = {},
                            ) -> List[MarketPublisherConfig]:
    """Simulates the best configuration based on the lowest CPA

    Args:
        market_id_publisher_config_record (pd.DataFrame): Data with configurations
        cpc_per_market_id_publisher_records (pd.DataFrame): Data with CPC values
        model_name (str, optional): Name of the model to use for embedding. Defaults to "all-MiniLM-L6-v2".
        top_k (int, optional): Number of top configurations to get. Defaults to 5.

    Returns:
        MarketPublisherConfig: Best configuration
    """

    configurations = []
    

    pred_input_data = None
    for row in configuration_search_space.iterrows():
        market_id = row[1]["market_id"]
        publisher = row[1]["publisher"]
        preprocessed_classifier_input = preprocess_input_for_conversion_classifier(customer_id_freq=customer_id_freq,
                                                                                              customer_id=customer_id,
                                                                                              category_id=category_id,
                                                                                              category_id_freq=category_id_freq,
                                                                                              market_id=market_id, 
                                                                                              market_id_freq=market_id_freq,
                                                                                              publisher_freq=publisher_freq, 
                                                                                              publisher_id=publisher,
                                                                                              cpc_per_market_id_publisher_records=cpc_per_market_id_publisher_records,
                                                                                              industry=industry,
                                                                                              values_embeddings_map=values_embeddings_map
                                                                                              
                                                                                              )
        pred_input_data = preprocessed_classifier_input["preprocessed_data"]
        preprocessed_predictors = preprocessed_classifier_input["preprocessed_predictors"]
        proba_to_convert = conversion_classifier.predict_proba(preprocessed_predictors)[:, 1]
        
        preprocessed_predictors = preprocess_input_for_cpa_model(proba_to_convert=proba_to_convert,
                                                                input_data=pred_input_data,
                                                                )
        
        cpa_pred_res = cpa_model.predict(preprocessed_predictors)[0]
        config = MarketPublisherConfig(market_id=market_id, publisher=publisher, 
                                       cpc=cpa_pred_res
                                       )
        configurations.append(config)
        return configurations


def predict_conversion_proba(classifier_model, predictors):
    proba = classifier_model.predict_proba(predictors)[:, 1]
    proba
    

def predict_cpa(model, predictors):
    predictions =model.predict(predictors)
    return predictions





if __name__ == "__main__":
    from utils import get_model, get_path, get_metadata
    import numpy as np
    
    conversion_model_path = get_path(folder_name='model_store', 
                                 file_name='conversion_classifier.model'
                                 )
    cpa_model_path = get_path(folder_name='model_store',
                            file_name='cpa.model'
                            )
    df_filepath = get_path(folder_name="", file_name="Data Science Challenge - ds_challenge_data.csv")
    df = pd.read_csv(df_filepath)
    classifier = get_model(conversion_model_path)
    cpa_regressor = get_model(cpa_model_path)


    metadata = get_metadata(customer_id_freq_filepath="metadata_store/customer_id_freq.csv",
                                category_id_freq_filepath="metadata_store/category_id_freq.csv",
                                market_id_freq_filepath="metadata_store/market_id_freq.csv",
                                publisher_id_freq_filepath="metadata_store/publisher_id_freq.csv",
                                cpc_per_market_id_publisher_records_filepath="metadata_store/cpc_per_market_id_publisher_records.csv"
                                )
    
    customer_id_freq_record = metadata.customer_id_freq_record
    category_id_freq_record = metadata.category_id_freq_record
    market_id_freq_record = metadata.market_id_freq_record
    publisher_freq_record = metadata.publisher_freq_record
    cpc_per_market_id_publisher_records = metadata.cpc_per_market_id_publisher_records
    
    unique_customer_ids = customer_id_freq_record["customer_id"].unique().tolist()
    unique_industries = df["industry"].unique().tolist()
    unique_category_ids = category_id_freq_record["category_id"].unique().tolist()
    unique_publisher_ids = publisher_freq_record["publisher"].unique().tolist()
    unique_market_ids = market_id_freq_record["market_id"].unique().tolist()


    random_customer_id  = np.random.choice(unique_customer_ids, 1)[0]

    random_industry = np.random.choice(unique_industries, 1)[0]

    random_category_id = np.random.choice(unique_category_ids, 1)[0]
    random_publisher_id = np.random.choice(unique_publisher_ids, 1)[0]
    random_market_id = np.random.choice(unique_market_ids, 1)[0]
    market_id_publisher_freq_config = (df.groupby(["market_id", "publisher"])["campaign_id"]
                                   .agg("count").reset_index()
                                   .sort_values("campaign_id", ascending=False)
                                   )
    from sentence_transformers import SentenceTransformer
    model_name: str = "all-MiniLM-L6-v2"
    
    top_20_configurations = get_top_k_configurations(market_id_publisher_freq_config, k=20)


    simulated_configurations = simulate_configuration(cpc_per_market_id_publisher_records=cpc_per_market_id_publisher_records,
                                                market_id_freq=market_id_freq_record,
                                                publisher_freq=publisher_freq_record,
                                                customer_id_freq=customer_id_freq_record,
                                                category_id_freq=category_id_freq_record,
                                                category_id=random_category_id,
                                                customer_id=random_customer_id,
                                                industry=random_industry,
                                                conversion_classifier=classifier,
                                                configuration_search_space=top_20_configurations,
                                                cpa_model=cpa_regressor,
                                                values_embeddings_map={},
                                                )
    recommend_config = recommend_configuration(simulated_configurations)
    print(recommend_config)
    