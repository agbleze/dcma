
from utils import get_path
import pandas as pd
import numpy as np
from inference import predict_cpa


# load metadata
customer_id_filepath = get_path("metadata_store", "customer_id_freq.csv")
category_id_freq_filepath = get_path("metadata_store", "category_id_freq.csv")
market_id_freq_filepath = get_path("metadata_store", "market_id_freq.csv")
publisher_id_freq_filepath = get_path("metadata_store", "publisher_id_freq.csv")
cpc_per_market_id_publisher_records_filepath = get_path("metadata_store", "cpc_per_market_id_publisher_records.csv")

customer_id_freq_df = pd.read_csv(customer_id_filepath)
category_id_freq_df = pd.read_csv(category_id_freq_filepath)
market_id_freq_df = pd.read_csv(market_id_freq_filepath)
publisher_id_freq_df = pd.read_csv(publisher_id_freq_filepath)
cpc_per_market_id_publisher_records = pd.read_csv(cpc_per_market_id_publisher_records_filepath)
df_path = get_path(folder_name="", file_name="Data Science Challenge - ds_challenge_data.csv")
df = pd.read_csv(df_path)

# sample inputs to use
unique_customer_ids = df["customer_id"].unique()
unique_industries = df["industry"].unique()
unique_category_ids = df["category_id"].unique()
unique_publisher_ids = df["publisher"]
unique_market_ids = df["market_id"].unique()
random_customer_id  = np.random.choice(unique_customer_ids, 1)[0]
random_industry = np.random.choice(unique_industries, 1)[0]
random_category_id = np.random.choice(unique_category_ids, 1)[0]
random_publisher_id = np.random.choice(unique_publisher_ids, 1)[0]
random_market_id = np.random.choice(unique_market_ids, 1)[0]
known_market_id, known_publisher_id = cpc_per_market_id_publisher_records[["market_id", "publisher"]].values[0]


known_market_id, known_publisher_id = cpc_per_market_id_publisher_records[["market_id", "publisher"]].values[0]
convert_model_path = get_path("model_store", "conversion_classifier.model")
cpa_model_path = get_path("model_store", "cpa.model")

    
if __name__ == "__main__":
    #### test with known_market_id and known_publisher_id and other random selected inputs
    cpa_pred = predict_cpa(convert_model_path=convert_model_path, cpa_model_path=cpa_model_path,
                           category_id=random_category_id, customer_id=random_customer_id,
                           publisher_id=known_publisher_id, industry=random_industry,
                           market_id=known_market_id,
                           customer_id_freq_record=customer_id_freq_df,
                           category_id_freq_record=category_id_freq_df,
                           market_id_freq_record=market_id_freq_df,
                           publisher_freq_record=publisher_id_freq_df,
                           cpc_per_market_id_publisher_records=cpc_per_market_id_publisher_records
                           )
    print(f"CPA prediction: {cpa_pred}")
    assert isinstance(cpa_pred[0], float) 









import json
from typing import Union


import os
import json
import numpy as np
from typing import Union

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
