
import pandas as pd
import numpy as np
from typing import Dict, Union
from sentence_transformers import SentenceTransformer

def get_encoded_values(encoder_data: pd.DataFrame, encoder_colname: str,
                       encoder_values_column: str,
                       value_to_encode: Union[str, int, float]
                       ):
    
    if value_to_encode not in encoder_data[encoder_colname].values:
        return 0
    encoded_values = encoder_data[encoder_data[encoder_colname]==value_to_encode][encoder_values_column].values[0]
    return encoded_values
    
def encode(data: pd.DataFrame, colname_to_encode, 
           save_encoded_column_as: str,
            encoder_data: pd.DataFrame,
            encoder_colname: str,
            encoder_values_colname: str
            ) -> pd.DataFrame:
    
    categories = data[colname_to_encode].unique()
    
    for category in categories:
        encode_value = get_encoded_values(encoder_data=encoder_data, 
                                            encoder_colname=encoder_colname,
                                            encoder_values_column=encoder_values_colname,
                                            value_to_encode=category
                                            )
        
        data.loc[data[colname_to_encode] == category, save_encoded_column_as] = encode_value
    return data


def get_values_to_embedding_map(values_to_encode: list,
                                model_name: str = "all-MiniLM-L6-v2"
                                ):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(values_to_encode, convert_to_numpy=True)
    values_embeddings_map = dict(zip(values_to_encode, embeddings))
    return values_embeddings_map


def get_embedding(value_to_embed, values_embeddings_map: dict, 
                  model: SentenceTransformer = None,
                  return_updated_values_embeddings_map: bool = False
                  ):
        if value_to_embed not in values_embeddings_map:
            if model is None:
                raise ValueError("Model must be provided to fetch embedding when value_to_embed is not available in values_embeddings_map.")
            embedding = model.encode(value_to_embed, convert_to_numpy=True)
            values_embeddings_map[value_to_embed] = embedding
        else:
            embedding = values_embeddings_map[value_to_embed]
        if not return_updated_values_embeddings_map:
            return embedding
        else:    
            return embedding, values_embeddings_map
    
def encode_with_embeddings(data: pd.DataFrame, 
                            colname_to_encode: str,
                            save_embedding_column_as: str,
                            values_embeddings_map: dict = {},
                            model_name: str = "all-MiniLM-L6-v2",
                            ) -> pd.DataFrame:
    """Encodes a categorical variable in the data using embeddings from a pre-trained model

    Args:
        data (pd.DataFrame): Data to be encoded
        colname_to_encode (str): Column name of the variable to encode
        save_embedding_column_as (str): Column name to save the embeddings
        model_name (str, optional): Name of the pre-trained model to use for encoding. Defaults to "all-MiniLM-L6-v2".

    Returns:
        pd.DataFrame: Data with additional column for encoded values
    """
    if colname_to_encode not in data.columns:
        raise ValueError(f"Data must contain {colname_to_encode} column to encode.")
    
    model = SentenceTransformer(model_name)
    data[save_embedding_column_as] = data[colname_to_encode].map(lambda v: get_embedding(v, values_embeddings_map, model))
    
    return data

def encode_multiple_data(data, variable_encoder_map: Dict,
                            encoder_values_colname
                            ):
    df = None
    for colname, encoder_df in variable_encoder_map.items():
        
        df = encode(data=data, colname_to_encode=colname,
                    save_encoded_column_as=f"{colname}_encoded",
                    encoder_data=encoder_df,
                    encoder_values_colname=encoder_values_colname, 
                    encoder_colname=colname
                    )
    return df

def transform_data_with_conversion(data: pd.DataFrame, variable_encoder_map: Dict,
                                    predictors, classifier, target="log_CPA",
                                    encoder_values_colname="count",
                                    colname_to_encode="industry",
                                    save_embedding_column_as="industry_embedding"
                                    ):
        data = encode_multiple_data(data=data,
                                    encoder_values_colname=encoder_values_colname,
                                    variable_encoder_map=variable_encoder_map,
                                    )
        data = encode_with_embeddings(data=data, colname_to_encode=colname_to_encode,
                                     save_embedding_column_as=save_embedding_column_as
                                     )
        processed_data = prepare_data(data=data, 
                                    predictors=predictors, 
                                    target=target
                                    )
        predictors = processed_data["predictors"]
        proba_to_convert = classifier.predict_proba(predictors)[:, 1].tolist()
        data["proba_convert"] = proba_to_convert
        data["proba_convert"] = data["proba_convert"].astype(float)
        return data
 
def prepare_data(data: pd.DataFrame, 
                predictors: list,
                target: Union[str,None]=None,
                embedding_colname: str = 'industry_embedding'
                ) -> tuple:
    embedding_data = np.vstack(data[embedding_colname].values.tolist())
    non_embedding_data = data[predictors].to_numpy()
    all_predictors = np.hstack([non_embedding_data, embedding_data])
    predictor_colnames_in_order = predictors.copy()
    predictor_colnames_in_order.append(embedding_colname)
    if target is None:
        target_data = None
        combined_data = all_predictors
        combined_columns = predictor_colnames_in_order
        colpos = [combined_columns.index(i) for i in predictors]
        embedding_colpos = [combined_columns.insert(embedding_colname)]
        for i in range(len(embedding_data.shape[1])):
            embedding_colpos.append(embedding_colpos[-1] + (i+1))
        colpos.append(embedding_colpos)
        
    else:
        target_data = data[target].values
    
        combined_data = np.hstack([target_data.reshape(-1,1), all_predictors])
        combined_columns = [target] + predictor_colnames_in_order
        colpos = [combined_columns.index(i)]
        colpos.extend([combined_columns.index(i) for i in predictors])
        embedding_colpos = [combined_columns.insert(embedding_colname)]
        for i in range(len(embedding_data.shape[1])):
            embedding_colpos.append(embedding_colpos[-1] + (i+1))
        colpos.append(embedding_colpos)
    
    return {"target": target_data,
            "predictors": all_predictors,
            "predictor_colnames_inorder": predictor_colnames_in_order,
            "full_data": combined_data,
            "full_data_columns_in_order": combined_columns,
            "full_data_columns_positions": colpos
            } 
           

def create_binary_conversion_variable(data: pd.DataFrame, 
                                      conversion_colname: str = "converions",
                                      save_binary_conversion_as: str = "convert"
                                      ) -> pd.DataFrame:
    """Creates a binary variable for conversions in the data

    Args:
        data (pd.DataFrame): Data to be analyzed
        conversion_colname (str, optional): Column name for conversions. Defaults to "converions".
        binary_conversion_colname (str, optional): Column name to save the binary conversion variable. Defaults to "convert".

    Returns:
        pd.DataFrame: Data with additional column for binary conversion variable
    """
    if conversion_colname not in data.columns:
        raise ValueError(f"Data must contain {conversion_colname} column to create binary conversion variable.")
    
    data[save_binary_conversion_as] = np.where((data[conversion_colname]==0), 0, 1)
    return data


def augment_data_with_conversion_proba(data: pd.DataFrame, 
                                        model, 
                                        predictors: list,
                                        embedding_colname: str = 'industry_embedding',
                                        save_proba_column_as: str = 'proba_convert'
                                        ) -> pd.DataFrame:
    """Augments data with predicted probability of conversion

    Args:
        data (pd.DataFrame): Data to be augmented
        model (_type_): Model to use for prediction
        predictors (list): List of predictor variables
        embedding_colname (str, optional): Column name for embeddings. Defaults to 'industry_embedding'.

    Returns:
        pd.DataFrame: Data with additional column for predicted probability of conversion
    """
    prepared_data = prepare_data(data=data, predictors=predictors,
                                target="convert",
                                embedding_colname=embedding_colname
                                )
    
    X_convert_data = prepared_data["predictors"]
    
    proba_to_convert = model.predict_proba(X_convert_data)[:, 1].tolist()
    data[save_proba_column_as] = proba_to_convert
    data[save_proba_column_as] = data[save_proba_column_as].astype(float)
    
    return data


def get_cpc_imputation(cpc_record: pd.DataFrame,
                       market_id: str,
                       publisher: str,
                       cpc_values_colname: str = "CPC",
                       ):
    """Gets the CPC imputation for a given market_id and publisher
    Args:
        cpc_record (pd.DataFrame): Data with CPC values
        market_id (str): Market ID to filter by
        publisher (str): Publisher to filter by
        cpc_values_colname (str, optional): Column name for CPC values. Defaults to "CPC".
    Returns:
        float: CPC imputation value
    """
    if cpc_values_colname not in cpc_record.columns:
        raise ValueError(f"Data must contain {cpc_values_colname} column to compute CPC imputation.")
    cpc_record["publisher"] = cpc_record["publisher"].astype(str)
    cpc_imputation = cpc_record[(cpc_record["market_id"]==market_id) & (cpc_record["publisher"]==publisher)][cpc_values_colname].values#[0]
    if cpc_imputation.size == 0:
        cpc_imputation = cpc_record[cpc_values_colname].median()
    else:
        cpc_imputation = cpc_imputation[0]
    return cpc_imputation
    