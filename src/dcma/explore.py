
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