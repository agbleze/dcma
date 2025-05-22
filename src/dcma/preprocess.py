import pandas as pd
import numpy as np
from transform import (encode, prepare_data, get_encoded_values, 
                       get_cpc_imputation, encode_with_embeddings
                       )
from typing import Dict,List, Union, NamedTuple, Literal
import os
from dataclasses import dataclass
from sklearn.utils.class_weight import compute_class_weight
import inspect

def create_frequency_table(data: pd.DataFrame, 
                            variable: str
                            ) -> pd.DataFrame:
    """Creates a frequency column for a categorical variable in the data

    Args:
        data (pd.DataFrame): Data to be analyzed
        variable (str): Variable to create frequency column for

    Returns:
        pd.DataFrame: Data with additional column for frequency of the variable
    """
    if variable not in data.columns:
        raise ValueError(f"Data must contain {variable} column to compute its frequency.")
    
    freq_table = data[variable].value_counts().reset_index()
    return freq_table

def calculate_stats_per_group(data: pd.DataFrame, target_var: str, 
                              groupby_var: Union[List,str],
                              stats_to_compute: str="mean"
                              ):
    if isinstance(groupby_var, str):
        groupby_var = [groupby_var]
    for var in groupby_var:
        if var not in data.columns:
            raise ValueError(f"{var} is not a valid column name. {var} must be a column in data")
    if target_var not in data.columns:
        raise ValueError(f"{target_var} is not a valid column in data")
    
    stat_table = data.groupby(groupby_var)[target_var].agg(stats_to_compute).reset_index()
    return stat_table


def compute_cpa(data, cpa_colname: str = "CPA") -> pd.DataFrame:
    """Computes the Cost Per Application (CPA) given a data with cost and conversions column

    Args:
        data (_type_): Data to be analyzed
        cpa_colname (str, optional): Column for computed CPA. Defaults to "CPA".

    Returns:
        pd.DataFrame: Data with additional column (cpa_colname) for computed CPA
    """
    if ("cost" not in data.columns) or ("converions" not in data.columns):
        raise ValueError("Data must contain 'cost' and 'converions' columns required for CPA calculation.")
    
    
    data[cpa_colname] = np.where(((data["cost"]!=0) & (data["converions"]==0)), data["cost"], 0)
    data[cpa_colname] = np.where(((data["cost"]!=0) & (data["converions"]!=0)), (data["cost"] / data["converions"]), data[cpa_colname])
    data[cpa_colname] = np.where(((data["cost"]==0) & (data["converions"]!=0)), 0, data[cpa_colname])

    median = data[cpa_colname].median()
    data[cpa_colname] = np.where(((data["cost"]==0) & (data["converions"]==0)), median, data[cpa_colname])
    return data


def compute_cpc(data: pd.DataFrame, cost_colname: str = "cost",
                click_colname: str = "clicks",
                cpc_colname: str = "CPC"
                ) -> pd.DataFrame:
    """Computes the Cost Per Click (CPC) from data with columns for 
        cost and clicks. 

    Args:
        data (pd.DataFrame): Data to use for computation
        cost_colname (str, optional): Name of column with cost values in the data . Defaults to "cost".
        click_colname (str, optional): Name of column with click values in the data. Defaults to "click".
        cpc_colname (str, optional): Column name to save the estimated Cost Per Click. Defaults to "CPC".

    Returns:
        pd.DataFrame: Data with additional column (cpc_colname) for estimated CPC
    """
    if (cost_colname not in data.columns) or (click_colname not in data.columns):
        raise ValueError(f"Data must contain {cost_colname} and {click_colname} columns required for CPA calculation.")
    
    data[cpc_colname] = data[cost_colname] / data[click_colname]
    data[cpc_colname] = np.where(((data[cost_colname]!=0) & (data[click_colname]==0)), data[cost_colname], data[cpc_colname])
    data[cpc_colname] = np.where(((data[cost_colname]==0) & (data[click_colname]==0)), 0, data[cpc_colname])
    return data



def compute_cpa_per_variable(data: pd.DataFrame, 
                            variable: str, 
                            target_variable: str = 'CPA'
                            ) -> pd.DataFrame:
    """Computes mean and median CPA per variable in the data

    Args:
        data (pd.DataFrame): Data to be analyzed
        variable (str): Variable to compute CPA per
        target_variable (str, optional): Target variable. Defaults to 'CPA'.

    Returns:
        pd.DataFrame: Data with additional columns for mean and median CPA per variable
    """
    if variable not in data.columns:
        raise ValueError(f"Data must contain {variable} column required for CPA calculation.")
    
    exp_df = data.groupby(variable)[target_variable].agg(["mean", "median"]).reset_index().sort_values("mean", ascending=False)
    
    df_long = pd.melt(exp_df, id_vars=variable, 
                        var_name='CPA_Statistics_type', 
                        value_name='statistics_value'
                        )
    return df_long


class CPAPredictionDataSchema(NamedTuple):
    colnames: List = ['CPC',
                    'category_id_freq_encoded',
                    'market_id_freq_encoded',
                    'customer_id_freq_encoded',
                    'publisher_freq_encoded',
                    'proba_convert',
                    'industry_embedding'
                    ]

class ConvertProbabilityDataSchema(NamedTuple):
    colnames = ['CPC',
                'category_id_freq_encoded',
                'market_id_freq_encoded',
                'customer_id_freq_encoded',
                'publisher_freq_encoded',
                'industry_embedding'
                ]
    
def preprocess_input_for_conversion_classifier(customer_id_freq: pd.DataFrame, customer_id: str, 
                                               category_id_freq: pd.DataFrame,category_id: str, 
                                               market_id_freq: pd.DataFrame, market_id: str,
                                               publisher_freq: pd.DataFrame, publisher_id: str, 
                                               cpc_per_market_id_publisher_records: pd.DataFrame,
                                               industry: str, values_embeddings_map: dict,
                                               data_schema: ConvertProbabilityDataSchema = ConvertProbabilityDataSchema()
                                               ) -> Dict:
    customer_id_encode = get_encoded_values(encoder_data=customer_id_freq, encoder_colname="customer_id",
                                            encoder_values_column="count",
                                            value_to_encode=customer_id,
                                            )
    category_id_encode = get_encoded_values(encoder_data=category_id_freq, encoder_colname="category_id",
                                            encoder_values_column="count",
                                            value_to_encode=category_id,
                                            )
    
    market_id_encode = get_encoded_values(encoder_data=market_id_freq, encoder_colname="market_id",
                                                encoder_values_column="count",
                                                value_to_encode=market_id,
                                                )
    publisher_encode = get_encoded_values(encoder_data=publisher_freq, encoder_colname="publisher",
                                        encoder_values_column="count",
                                        value_to_encode=publisher_id
                                        )
    cpc_imp = get_cpc_imputation(cpc_record=cpc_per_market_id_publisher_records,
                                market_id=market_id,
                                publisher=publisher_id
                                )
    predictors = [predictor for predictor in data_schema.colnames if predictor != 'industry_embedding']
    data_vals = [cpc_imp, category_id_encode, market_id_encode, 
                customer_id_encode, publisher_encode, industry
                ]
    data_colnames = predictors.copy()
    data_colnames.append("industry")
    input_data = pd.DataFrame(data=[data_vals], columns=data_colnames)
    input_data = encode_with_embeddings(data=input_data, colname_to_encode='industry',
                                        save_embedding_column_as='industry_embedding',
                                        values_embeddings_map=values_embeddings_map,
                                        model_name="all-MiniLM-L6-v2"
                                        )
    # ensure data is in the correct order to transform for conversion probability prediction
    input_data = input_data[data_schema.colnames]
    
    inference_prepared_data = prepare_data(data=input_data, predictors=predictors,
                                            target=None,
                                            embedding_colname="industry_embedding"
                                            )
    preprocessed_predictors = inference_prepared_data["predictors"]
    return {"preprocessed_data": input_data,
            "preprocessed_predictors": preprocessed_predictors
            }

def preprocess_input_for_cpa_model(proba_to_convert: Union[float,List[float]],
                                   input_data: pd.DataFrame,
                                   predictors_data_schema: CPAPredictionDataSchema = CPAPredictionDataSchema() 
                                   ):
    input_data["proba_convert"] = proba_to_convert
    # to ensure colunmns are in expected order
    pred_input_data = input_data[predictors_data_schema.colnames] 
    predictors = [predictor for predictor in predictors_data_schema.colnames if predictor != "industry_embedding"]
    preprocessed_data = prepare_data(data=pred_input_data, predictors=predictors,
                                    target=None,
                                    embedding_colname="industry_embedding"
                                    )
    preprocessed_predictors = preprocessed_data["predictors"]
    return preprocessed_predictors


@dataclass
class FeatureEncoderStore:
    pass

class PreprocessedDataStore(NamedTuple):
    predictors: Union[np.array, pd.DataFrame]
    target: Union[np.array, pd.DataFrame]
    predictor_colnames_inorder: List
    full_data: np.array
    full_data_columns_in_order: List
    
class PreprocessPipeline(object):
    def __init__(self, data: pd.DataFrame, categorical_features: List,
                 numeric_features: List,
                 features_to_embed: List,
                 target_variable: str
                 ):
        self.data = data
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.features_to_embed = features_to_embed
        self.target_variable = target_variable
        
    def create_encoder_records(self, #categorical_features: List,
                                encoder_type: Literal["target_encoding", "frequency_encoding"],
                                stats_to_compute: str = "mean",
                                data: Union[pd.DataFrame, None] = None
                                ):
        if not isinstance(data, pd.DataFrame):
            data = self.data
        # done on purpose to ensure later use of this field column name for encoding
        self.stats_to_compute = stats_to_compute 
        if encoder_type not in ["target_encoding", "frequency_encoding"]:
            raise NotImplementedError(f"{encoder_type} is not a valid implemented encoder")
        
        self.feat_encoder_store = FeatureEncoderStore()
        if encoder_type == "frequency_encoding":
            for catvar in self.categorical_features:
                cat_encode_record = create_frequency_table(data=data, variable=catvar)
                setattr(self.feat_encoder_store, catvar, cat_encode_record)
                self.encoder_values_colname = cat_encode_record.columns[-1]
        elif encoder_type == "target_encoding":
            for catvar in self.categorical_features:
                cat_encode_record = calculate_stats_per_group(data=data, 
                                                              target_var=self.target_variable,
                                                              groupby_var=catvar,
                                                              stats_to_compute=stats_to_compute
                                                              )
                setattr(self.feat_encoder_store, catvar, cat_encode_record)
                self.encoder_values_colname = cat_encode_record.columns[-1]
    
    def export_encoder_records(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if not hasattr(self, "feat_encoder_store"):
            raise ValueError("records for encoding have not been create hence not exportable")
        else:
            for catvar in self.categorical_features:
                catvar_record = getattr(self.feat_encoder_store, catvar)
                catvar_record.to_csv(os.path.join(save_dir, f"{catvar}.csv"))
    
    
    def create_cpc_feature(self):
        self.data = compute_cpc(data=self.data)            
    
    def encode_features(self, encoder_type="target_encoding",
                        stats_to_compute: str = "mean",
                        data: Union[pd.DataFrame, None] = None,
                        encoder_values_colname: Union[str,None] = None,
                        **kwargs
                        ):
        if not isinstance(data, pd.DataFrame):
            data = self.data
        self.predictors = []
        if not hasattr(self, "feat_encoder_store"):
            # done on purpose to ensure it can used in the encoding process
            self.stats_to_compute = stats_to_compute
            print(f"feat_encoder_store does not exist. ... creating them for encoding")
            _kwarg_create_encoder_records = inspect.signature(self.create_encoder_records).parameters
            _passed_create_encoder_records_params = {k:v for k, v in kwargs.items() 
                                                     if k in _kwarg_create_encoder_records.keys()
                                                     }
            #self.feat_encoder_store = 
            self.create_encoder_records(encoder_type=encoder_type,
                                        stats_to_compute=self.stats_to_compute,
                                        **_passed_create_encoder_records_params
                                        )
            
            _kwarg_export_encoder_records = inspect.signature(self.export_encoder_records).parameters
            _passed_export_encoder_records_params = {k:v for k, v in kwargs.items() 
                                                     if k in _kwarg_export_encoder_records.keys()
                                                     }
            self.export_encoder_records(**_passed_export_encoder_records_params)
        else:
            print(f"Using already exising feat_encoder_store for encoding")
        if not encoder_values_colname:
            encoder_values_colname = self.encoder_values_colname
        encoded_categorical_var = []
        for catvar in self.categorical_features:
            encoded_colname = f'{catvar}_encoded'
            catvar_encode_record = getattr(self.feat_encoder_store, catvar)
            self.data = encode(data=data, colname_to_encode=catvar,
                                save_encoded_column_as=encoded_colname,
                                encoder_data=catvar_encode_record,
                                encoder_colname=catvar,
                                encoder_values_colname=encoder_values_colname,
                                )
            encoded_categorical_var.append(encoded_colname)
        self.predictors.extend(encoded_categorical_var)
        if not isinstance(self.numeric_features, list):
            self.predictors.append(self.numeric_features)
        else:
            self.predictors.extend(self.numeric_features)   
        return self.data
                
    def transform_columns_to_embed(self, 
                                   data: Union[pd.DataFrame, None]=None,
                                   values_embeddings_map: Dict={}
                                   ):
        if not isinstance(data, pd.DataFrame):
            data = self.data
        self.embedding_colname = []
        if isinstance(self.features_to_embed, str):
            self.features_to_embed = [self.features_to_embed]
        for feat_name in self.features_to_embed:
            save_embedding_column_as=f"{feat_name}_embedding"
            self.data = encode_with_embeddings(data=data, colname_to_encode=feat_name,
                                                save_embedding_column_as=save_embedding_column_as,
                                                values_embeddings_map=values_embeddings_map
                                                )
            self.embedding_colname.append(save_embedding_column_as)
        return self.data
            
    def prepare_modelling_data(self, predictors, embedding_colname, 
                               data: Union[pd.DataFrame, None]=None,
                               target: Union[str, None] = None
                               )->PreprocessedDataStore:
        if not isinstance(data, pd.DataFrame):
            data = self.data
        preprocessed_input_data = prepare_data(data=data, predictors=predictors,
                                                target=target,
                                                embedding_colname=embedding_colname
                                                )
        self.preprocessed_predictor_data = preprocessed_input_data["predictors"]
        self.preprocessed_target_data = preprocessed_input_data["target"]
        self.predictor_colnames_inorder = preprocessed_input_data["predictor_colnames_inorder"]
        self.full_data = preprocessed_input_data["full_data"]
        self.full_data_columns_in_order = preprocessed_input_data["full_data_columns_in_order"]
        return PreprocessedDataStore(predictors=self.preprocessed_predictor_data,
                                     target=self.preprocessed_target_data,
                                     predictor_colnames_inorder=self.predictor_colnames_inorder,
                                     full_data=self.full_data,
                                     full_data_columns_in_order=self.full_data_columns_in_order
                                     )
    
    def compute_sample_weights(self, categorical_target):
        _target_data = self.data[categorical_target] 
        classes = _target_data.unique()
        class_weights = compute_class_weight(class_weight="balanced", 
                                             classes=classes, 
                                             y=_target_data
                                             )
        self.class_weight_dict = dict(zip(classes, class_weights))

        self.sample_weight = np.array([self.class_weight_dict[label] for label in _target_data])
        
    def run_preprocess_pipeline(self, categorical_target,
                                encoder_type: Literal["target_encoding", "frequency_encoding"],
                                save_dir: str, stats_to_compute: str = "mean",
                                predictors=None, 
                                embedding_colname=None,
                                target=None,
                                cal_sample_weights: bool = True,
                                make_modelling_data: bool = True,
                                **kwargs
                                )->PreprocessedDataStore:
        os.makedirs(save_dir, exist_ok=True)
        if cal_sample_weights:
            self.compute_sample_weights(categorical_target=categorical_target) 
        # self.create_encoder_records(encoder_type=encoder_type,
        #                             stats_to_compute=stats_to_compute
        #                             )   
        # self.export_encoder_records(save_dir=save_dir) 
        _kwarg_encode_features = inspect.signature(self.encode_features).parameters
        _passed_encode_features_params = {k:v for k, v in kwargs.items() 
                                        if k in _kwarg_encode_features.keys()
                                        }
        self.encoded_data = self.encode_features(encoder_type=encoder_type,
                                                 save_dir=save_dir,
                                                 stats_to_compute=stats_to_compute, 
                                                 **kwargs #**_passed_encode_features_params
                                                 )   
        self.encoded_embedded_data = self.transform_columns_to_embed(data=self.encoded_data) 
        if not predictors:
            predictors = self.predictors
        if not embedding_colname:
            embedding_colname = self.embedding_colname
        if not target:
            target = self.target_variable
            
        if not make_modelling_data:
            preprocessed_data_store = PreprocessedDataStore(predictors=self.encoded_embedded_data[predictors],
                                                            target=self.encoded_embedded_data[target],
                                                            predictor_colnames_inorder=predictors.copy().append(embedding_colname)
                                                            )
        if make_modelling_data:
            preprocessed_data_store = self.prepare_modelling_data(predictors=predictors,
                                                                embedding_colname=embedding_colname,
                                                                target=target
                                                                )  
            return preprocessed_data_store     




def main():
    pass