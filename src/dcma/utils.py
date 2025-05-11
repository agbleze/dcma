from typing import NamedTuple
import pandas as pd
import os
import joblib


def get_path(folder_name: str, file_name: str):
    dirpath = os.getcwd()
    filepath = f'{dirpath}/{folder_name}/{file_name}'
    return filepath

def get_number_of_unique_values(data: pd.DataFrame, variable: str):
    num_values = data[variable].nunique()
    print(f'{variable} has {num_values} unique values')

class MarketPublisherConfig(NamedTuple):
  market_id: str
  publisher: str
  cpc: float

class MetaDataRecords(NamedTuple):
    customer_id_freq_record: pd.DataFrame
    category_id_freq_record: pd.DataFrame
    market_id_freq_record: pd.DataFrame
    publisher_freq_record: pd.DataFrame
    cpc_per_market_id_publisher_records: pd.DataFrame
    
    
def get_metadata(customer_id_freq_filepath: str, 
                 category_id_freq_filepath: str,
                 market_id_freq_filepath: str,
                 publisher_id_freq_filepath: str,
                 cpc_per_market_id_publisher_records_filepath: str
                 )->MetaDataRecords:
    customer_id_freq_df = pd.read_csv(customer_id_freq_filepath)
    category_id_freq_df = pd.read_csv(category_id_freq_filepath)
    market_id_freq_df = pd.read_csv(market_id_freq_filepath)
    publisher_id_freq_df = pd.read_csv(publisher_id_freq_filepath)
    cpc_per_market_id_publisher_records = pd.read_csv(cpc_per_market_id_publisher_records_filepath)
    metadata = MetaDataRecords(customer_id_freq_record=customer_id_freq_df,
                                category_id_freq_record=category_id_freq_df,
                                market_id_freq_record=market_id_freq_df,
                                publisher_freq_record=publisher_id_freq_df,
                                cpc_per_market_id_publisher_records=cpc_per_market_id_publisher_records
                                )
    return metadata
    

   
def get_missing_data_percent(data: pd.DataFrame, variable: str) -> None:
    """Estimates percentage of missing data in a variable (column)
    in a pandas dataframe. The function takes a pandas dataframe and
    a variable name as input and prints the percentage of missing data

    Args:
        data (pd.DataFrame): Data to be analyzed
        variable (str): variable in the data to analyze for missing data
    """
    total_missing = data[variable].isnull().sum()
    total_data = data.shape[0]
    percent_missing = (total_missing / total_data) * 100
    print(f'Percentage of data missing in {variable}: {round(percent_missing, 2)}%')
    
    
def show_cross_validation_metrics(cv_result: dict, metrics: list):
    for metric in metrics:
        if metric not in list(cv_result.keys()):
            raise ValueError(f"{metric} is not in cv_result. cv_result metrics are {list(cv_result.keys())}")
        print(f"mean {metric}: {cv_result[metric].mean()}")



def get_model(model_path: str):
    model = joblib.load(model_path)
    return model