
#%%
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier
import sklearn.ensemble as ensembel
import sklearn.linear_model as lm
import sklearn.neighbors as nb
import inspect
from sklearn import naive_bayes

#%% 
albu_methods = [method for method in dir(lm) if callable(getattr(lm, method)) 
                and not method.startswith("__") and hasattr(getattr(lm, method), "fit")
                ]

#%%

nb_methods = [method for method in dir(nb) if callable(getattr(nb, method)) 
                and not method.startswith("__") and hasattr(getattr(nb, method), "fit")
                ]


#%%
[method for method in dir(naive_bayes) if callable(getattr(naive_bayes, method)) 
                and not method.startswith("__") and hasattr(getattr(naive_bayes, method), "fit")
                ]
#%%

albu_methods.extend(nb_methods)

#%%
albu_methods


#%%
regressors = []
for i in albu_methods:
    if "Regressor" in i:
        regressors.append(i)
        


#%%

getattr(lm, 'LinearRegression')


#%%
import sklearn 
hasattr(getattr(sklearn, 'LinearRegression'), "fit")

#%%

inspect.signature(getattr(ensembel,'HistGradientBoostingRegressor').fit).parameters


#%%

if "sample_weight" in inspect.signature(getattr(ensembel,'HistGradientBoostingRegressor').fit).parameters:
    print("yes")
else:
    print("No")

#%%
for i in albu_methods:
    if hasattr(getattr(lm, i), "fit"):
        print(i)
    else:
        print(f"{i} has no fit method")


# %%
invalid_params = [param for param in aug_params 
                              if param not in inspect.signature(getattr(A, augtype)).parameters
                              ]



#%%
import numpy as np
import pandas as pd
from typing import Union

def prepare_data(
    data: pd.DataFrame, 
    predictors: list,
    target: Union[str, None] = None,
    embedding_colname: str = 'industry_embedding'
) -> dict:
    """
    Prepares data by:
    - Processing the embedding column into a NumPy array.
    - Combining non-embedding predictors.
    - If a target is provided, reshaping it and concatenating it
      as the first column of the resulting array.
      
    Returns a dictionary with:
      - "combined": NumPy array where the first column is the target (if provided)
                    followed by all predictors.
      - "columns_in_order": List of column names corresponding to the combined array.
    """
    # Convert the embedding column, which is assumed to be in list format, into a 2D array.
    embedding_data = np.vstack(data[embedding_colname].values.tolist())
    
    # Convert the predictor columns (excluding the embedding) into a NumPy array.
    non_embedding_data = data[predictors].to_numpy()
    
    # Combine non-embedding data with embeddings.
    all_predictors = np.hstack([non_embedding_data, embedding_data])
    predictor_columns = predictors.copy()
    predictor_columns.append(embedding_colname)
    
    if target is not None:
        # Extract and reshape the target so it can concatenate horizontally.
        target_data = data[target].values.reshape(-1, 1)
        # Combine target with predictors. Target is placed as the first column.
        combined_data = np.hstack([target_data, all_predictors])
        combined_columns = [target] + predictor_columns
    else:
        combined_data = all_predictors
        combined_columns = predictor_columns
        
    return {
        "combined": combined_data,
        "columns_in_order": combined_columns
    }

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'industry_embedding': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'target': [0, 1, 0]
    })
    
    predictors = ['feature1', 'feature2']
    prepared = prepare_data(df, predictors, target='target')
    
    print("Combined Array:\n", prepared['combined'])
    print("Columns in Order:\n", prepared['columns_in_order'])



# %%
prepared['combined'][0]

#%%
pd.DataFrame(data=prepared['combined'], columns=prepared['columns_in_order'])
# %%
prepared['combined'][:, [1,2]]
# %%
sel = [i for i in prepared['columns_in_order'] if (i != prepared['columns_in_order'][-1])]
# %%
feat_indexes = [prepared['columns_in_order'].index(i) for i in sel ]
# %%

prepared['combined'][:, feat_indexes]
# %%
df_noembed = pd.DataFrame(data=prepared['combined'][:, feat_indexes], columns=sel)
# %%
df_noembed[prepared['columns_in_order'][-1]] = prepared['combined'][:, prepared['columns_in_order'].index(prepared['columns_in_order'][-1]):].tolist()
# %%
#%%
from minio import Minio

client = Minio(endpoint="127.0.0.1:9000",#"localhost:9000", 
               access_key="minioadmin", 
                secret_key="minioadmin",
                secure=False
                )



#%%
#client.bucket_exists("rawdata")


# %%
client.list_buckets()
# %%

#client.remove_bucket("rawdata")


# %%
bucket = client.list_buckets()[0]
# %%
bucket
# %%
objs = client.list_objects(bucket_name="rawdata", recursive=True, 
                           include_user_meta=True, include_version=True,
                           fetch_owner=True
                           )
# %%
objnms = [i.metadata for i in objs]
for i in objs:
    print(i.metadata)
# %%
client.stat_object(bucket_name="rawdata", 
                   object_name="test_1797afda-997d-4761-a78a-6d7ee0a78bb7.csv"
                   )
# %%
list(objs)
# %%
objnms
# %%
from data_ingest import (get_bucket_records, download_from_minio, 
                         get_minio_client,
                         get_expected_params_for_func
                         )

# %%
#get_minio_client()
bucket_records = get_bucket_records(minio_client=client, bucket_name="rawdata")
# %%
bucket_records[0].metadata

#%%

df = download_from_minio(minio_client=client, bucket_name="rawdata",
                    object_name=bucket_records[0].object_name
                    )
                    


# %%
df
# %%
[bc.object_name for bc in bucket_records if bc.metadata["uid"]== "1797afda-997d-4761-a78a-6d7ee0a78bb7"]

# %%
bucket_records[0].metadata["uid"]
# %%

prep_bucket_records = get_bucket_records(minio_client=client, 
                                      bucket_name="conversion-preprocess-storage"
                                      )

obj_nms = [bc.object_name for bc in prep_bucket_records if bc.metadata["uid"]== "65e46944-01bc-4e72-826d-258f04032275"]
# %%
test_prep_df = download_from_minio(minio_client=client,
                              bucket_name="conversion-preprocess-storage",
                              object_name=obj_nms[0], dytpe="npz"
                              )

train_prep_df = download_from_minio(minio_client=client,
                              bucket_name="conversion-preprocess-storage",
                              object_name=obj_nms[1], dytpe="npz"
                              )
# %%
train_prep = train_prep_df.get("preprocessed_data")
# %%
y_train = train_prep[:, 0]

#%%

for i in y_train:
    print(int(i))
# %%
X_train = train_prep[:, 1:]

# %%
X_train.shape
# %%

column_order = [bc.metadata.get("column_order") for bc in prep_bucket_records if bc.metadata["uid"]== "65e46944-01bc-4e72-826d-258f04032275"]
# %%
list(column_order[0])
# %%
import re, ast


ast.literal_eval(column_order[0].split("[")[0])
# %%
import re, ast

def parse_and_flatten(s: str) -> list:
    """
    Splits a comma-separated string on commas that lie outside square brackets,
    and then parses and flattens any token that is a list literal.
    
    For example, given:
      "convert,category_id_encoded,market_id_encoded,customer_id_encoded,publisher_encoded,CPC,['industry_embedding']"
    it returns:
      ['convert', 'category_id_encoded', 'market_id_encoded', 'customer_id_encoded', 'publisher_encoded', 'CPC', 'industry_embedding']
    """
    tokens = [t.strip() for t in re.split(r',(?=(?:[^[]*\[[^]]*])*[^]]*$)', s)]
    return sum(
        [ast.literal_eval(token) if token.startswith('[') and token.endswith(']') else [token]
         for token in tokens],
        []
    )

# Example usage:
input_str = "convert,category_id_encoded,market_id_encoded,customer_id_encoded,publisher_encoded,CPC,['industry_embedding']"
output = parse_and_flatten(input_str)
print(output)

# %%
import numpy as np
# %%
def func(name):
    B = name
    
print(func("b"))

# %%
