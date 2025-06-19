from preprocess import (compute_cpa, compute_cpc,
                        PreprocessPipeline
                        )
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Literal
from sklearn.model_selection import (train_test_split, 
                                     RandomizedSearchCV,
                                     cross_validate
                                     )
import sklearn.metrics as metrics
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn import ensemble, linear_model, neighbors
import inspect
import joblib
import json
import os
from transform import (create_binary_conversion_variable,
                       transform_data_with_conversion
                       )
from utils import show_cross_validation_metrics, get_path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class ModelTrainer(object):
    all_model_types = []
    algo_family = [ensemble, linear_model, neighbors]
    num_algo_family = len(algo_family)
    for algo in algo_family:
        # methods = [method for method in dir(algo) if callable(getattr(algo, method)) and not method.startswith("__") and hasattr(getattr(algo, method), "fit")
        #             ]
        for method in dir(algo):
            if callable(getattr(algo, method)) and not method.startswith("__") and hasattr(getattr(algo, method), "fit"):
            
                all_model_types.append(method)
        
    def __init__(self, training_predictors, training_target,
                  testing_predictors, testing_target,
                  sample_weight, model_type: str
                  ):
         self.training_predictors = training_predictors
         self.training_target = training_target
         self.testing_predictors = testing_predictors
         self.testing_target = testing_target
         self.sample_weight = sample_weight
         
         if model_type not in self.__class__.all_model_types:
             raise ValueError(f"{model_type} is not a valid sklearn model class name ")
         
         self.model_type = model_type
         
    def train_model(self, 
                    cv: int = 5,
                    scoring: Union[str, List]=['accuracy', "precision", "recall", "f1"],
                    ):
        include_sample_weight = False
        for algo_index, algo in enumerate(self.__class__.algo_family):
            #if (algo_index + 1) < self.__class__.num_algo_family:
                try:
                    model = getattr(algo, self.model_type)
                    model_params_signature = inspect.signature(getattr(algo, self.model_type).fit).parameters
                    if "sample_weight" in model_params_signature:
                        include_sample_weight = True
                    #algo_index = self.__class__.num_algo_family
                    break
                except Exception as e:
                    if (algo_index + 1) < self.__class__.num_algo_family:
                        continue
                    else:
                        raise ValueError(f"self.model_type not found in {self.algo_family}. Failed with error {e}")
                
        if "random_state" in model_params_signature:
            model = model(random_state=2025)
        else:
            model = model()
        if cv == 1:
            if include_sample_weight:
                print(f"sample weight used: {include_sample_weight}")
                model.fit(X=self.training_predictors, y=self.training_target, 
                            sample_weight=self.sample_weight
                            )
            else:
                print(f"sample weight used: {include_sample_weight}")
                model.fit(X=self.training_predictors, y=self.training_target)
        elif cv > 1:
            if include_sample_weight:
                params = {"sample_weight": self.sample_weight}
            else:
                params = None
            print(f"sample weight used: {include_sample_weight}")
            self.cv_results = cross_validate(estimator=model,
                                            X=self.training_predictors,
                                            y=self.training_target,
                                            cv=cv, n_jobs=-1,
                                            scoring=scoring,
                                            return_train_score=True, 
                                            return_estimator=True,
                                            params=params
                                            )
        self.estimator = self.cv_results["estimator"][0]
        return self.cv_results

    def show_model_results(self, model_result_metrics):
        if not hasattr(self, "cv_results"):
            print(f"cv_results not available ... Training model to get cv_results")
            self.cv_results = self.train_model()
        show_cross_validation_metrics(cv_result=self.cv_results,
                                      metrics=model_result_metrics
                                      )
    def evaluate_model(self, evaluation_metric="root_mean_squared_error"):
        valid_metrics = [method for method in dir(metrics) if 
                         callable(getattr(metrics, method)) and 
                         not method.startswith("__")
                         ]

        if not hasattr(self, "estimator"):
            print("Model has not been train hence estimator is not available ... Training model")
            self.cv_results = self.train_model()
            self.estimator = self.cv_results["estimator"][0]
            
        if evaluation_metric not in valid_metrics:
            raise ValueError(f"{evaluation_metric} is not a valid metric in sklearn metrics")
        metric_func = getattr(metrics, evaluation_metric, None)
        self.test_prediction = self.estimator.predict(self.testing_predictors)
        self.train_prediction = self.estimator.predict(self.training_predictors)
        
        self.test_score = metric_func(y_true=self.testing_target, y_pred=self.test_prediction) 
        self.train_score = metric_func(y_true=self.training_target, y_pred=self.train_prediction) 
        print(f'Model training {evaluation_metric} is: {round(self.train_score, 5)}')
        print(f'Model testing {evaluation_metric} is: {round(self.test_score, 5)}')
        
    def save_model(self, save_model_as: str, save_dir: str = "model_store"):
        if not hasattr(self, "estimator"):
            raise ValueError(f"No trained model available. Model needs to be trained before it can be saved")
        os.makedirs(save_dir, exist_ok=True)
        self.model_path = os.path.join(save_dir, save_model_as)
        joblib.dump(self.estimator, self.model_path)
        
    def save_model_metrics(self, save_dir: str = "model_store", 
                           save_model_metrics_as: Union[str, None] = None
                           ):
        if not hasattr(self, "cv_results"):
            raise ValueError(f"No trained model metrics available. Model needs to be trained for metrics to be saved")
        if not save_model_metrics_as:
            if hasattr(self, "model_path"):
                self.model_metrics_path = self.model_path.split(".")[0] + "_metrics.json"
            else:
                raise ValueError(f"Provide value for save_model_metrics_as since model_path is not known")
        else:
            self.model_metrics_path = os.path.join(save_dir, save_model_metrics_as)
        self.cv_metrics = {k: v for k, v in self.cv_results.items() if k != "estimator"}
        with open(self.model_metrics_path, "w") as file:
            json.dump(self.cv_metrics, file, cls=NpEncoder)
        print(f"successfully saved model metrics at {self.model_metrics_path}")
        
    def load_model(self, model_path: str):
        if not model_path:
            model_path = self.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path does not exist or inaccessible. Ensure model is trained and persisted at that {model_path}")
        model = joblib.load(model_path)
        return model
    
    def run_model_training_pipeline(self, cv, scoring, model_result_metrics,
                                    save_model_as,
                                    evaluation_metric="root_mean_squared_error",
                                    save_dir: str = "model_store"
                                    ):
        self.train_model(cv=cv, scoring=scoring)
        self.show_model_results(model_result_metrics=model_result_metrics)
        self.evaluate_model(evaluation_metric=evaluation_metric)
        self.save_model(save_model_as=save_model_as,
                        save_dir=save_dir
                        )
        self.save_model_metrics()
   
    def tune_model_hyperparameter(self, hyperparameter_space: Dict,
                                  n_iter: int,
                                  cv: int, scoring: str
                                  ):
        if not hasattr(self, "estimator"):
            raise ValueError(f"Model needs to be trained before tuning")
        self.random_search = RandomizedSearchCV(estimator=self.estimator, 
                                                param_distributions=hyperparameter_space,
                                                n_iter=n_iter,
                                                cv=cv,
                                                refit=True,
                                                return_train_score=True, 
                                                n_jobs=-1,
                                                random_state=2025,
                                                scoring=scoring,
                                                )  
        self.random_search.fit(X=self.training_predictors, y=self.training_target)     
        
      
if __name__ == "__main__":
    # read data
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    data_filepath = get_path(folder_name="", file_name="Data Science Challenge - ds_challenge_data.csv")
    data = pd.read_csv(data_filepath)
    # create target vars and predictors that are derieve from only other predictors
    data = compute_cpa(data=data)
    data = create_binary_conversion_variable(data)
    data = compute_cpc(data=data)
    
    # split into train and test
    target_convert_df = data["convert"]
    predictor_convert_df = data
    X_train_convert, X_test_convert, y_train_convert, y_test_convert = train_test_split(predictor_convert_df, 
                                                                                        target_convert_df, 
                                                                                        test_size=0.2, 
                                                                                        random_state=2025,
                                                                                        stratify=target_convert_df
                                                                                        )
    y_train_cpa = np.log1p(X_train_convert["CPA"].copy())
    y_test_cpa = np.log1p(X_test_convert["CPA"].copy())
    train_cpa = X_train_convert.copy()
    test_cpa = X_test_convert.copy()
    train_cpa["log_CPA"] = np.log1p(train_cpa["CPA"])
    test_cpa["log_CPA"] = np.log1p(test_cpa["CPA"])
   
    X_train_convert.drop(columns=["CPA"], inplace=True)
    X_test_convert.drop(columns=["CPA"], inplace=True)
    
    # prepare data for model
    categorical_features = ['category_id', 'market_id',
                            'customer_id', 'publisher',
                            ]
    numeric_features = ["CPC"]
    features_to_embed = ["industry"]
    preprocess_pipeline = PreprocessPipeline(data=X_train_convert, 
                                             categorical_features=categorical_features,
                                            numeric_features=numeric_features,
                                            target_variable="convert",
                                            features_to_embed="industry"
                                            )
    print("############  preprocessing  ###########")
    preprocessed_train_datastore = preprocess_pipeline.run_preprocess_pipeline(categorical_target="convert",
                                                                               encoder_type="frequency_encoding",
                                                                               save_dir="metadata_store",
                                                                               stats_to_compute="count"
                                                                              )
    preprocessed_train_target_convert = preprocessed_train_datastore.target
    preprocessed_train_predictors_convert = preprocessed_train_datastore.predictors
    preprocessed_train_predictor_colnames_inorder = preprocessed_train_datastore.predictor_colnames_inorder
    sample_weight = preprocess_pipeline.sample_weight
    X_train_encoded_embedded_data = preprocess_pipeline.encoded_embedded_data
    
    ### use the encoders created for train dataset to encode test dataset 
    # and prepare it for model evaluation. This prevents data leakage and 
    # ensures model evaluation reflect model performance in terms of the encoding 
    # used during training
    X_test_convert_encoded = preprocess_pipeline.encode_features(data=X_test_convert,
                                                                 stats_to_compute="count"
                                                                 ) 
    X_test_convert_encoded_embed = preprocess_pipeline.transform_columns_to_embed(data=X_test_convert_encoded)
    preprocess_test_datastore = preprocess_pipeline.prepare_modelling_data(predictors=preprocess_pipeline.predictors,
                                                                            embedding_colname=preprocess_pipeline.embedding_colname,
                                                                            data=X_test_convert_encoded_embed,
                                                                            target=preprocess_pipeline.target_variable
                                                                            )
    
    preprocessed_test_predictors_convert = preprocess_test_datastore.predictors
    preprocessed_test_target_convert = preprocess_test_datastore.target
    preprocessed_test_predictor_colnames_inorder = preprocess_test_datastore.predictor_colnames_inorder
    
    
    
    #  train conversion proba
    print("############# running model  ##########")
    trainer = ModelTrainer(training_predictors=preprocessed_train_predictors_convert,
                            training_target=preprocessed_train_target_convert,
                            testing_predictors=preprocessed_test_predictors_convert,
                            testing_target=preprocessed_test_target_convert,
                            sample_weight=sample_weight,
                            model_type="KNeighborsClassifier"
                            )
    trainer.run_model_training_pipeline(cv=20, scoring=['accuracy', "precision", "recall", "f1"],
                                        model_result_metrics=['test_accuracy',  'train_accuracy',
                                                              'test_precision', 'train_precision',
                                                              'test_recall', 'train_recall',
                                                              'test_f1', 'train_f1'
                                                              ],
                                        evaluation_metric="accuracy_score",
                                        save_model_as="conversion_proba.model",
                                        save_dir="model_store"
                                        )

    ##############   use conversion proba to augment data  #############
    classifier = trainer.estimator
    
    colnames_to_encode = ["market_id", "category_id", "customer_id", "publisher"]
    feat_encoder_dict = {col: getattr(preprocess_pipeline.feat_encoder_store, col) for col in categorical_features}
    
    train_cpa = transform_data_with_conversion(data=train_cpa, 
                                               variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )
    test_cpa = transform_data_with_conversion(data=test_cpa, variable_encoder_map=feat_encoder_dict,
                                                predictors=preprocess_pipeline.predictors,
                                                classifier=classifier
                                                )

   
    ############################################  CPA PREDITION    ##################################################
    ###############      prepare data for cpa proba   ##################
    ### use target encoding for cpa prediction using the median as of log_CPA for imputation of categorical var
    cpa_preproc_pipeline = PreprocessPipeline(data=train_cpa, 
                                                categorical_features=categorical_features,
                                                numeric_features=["CPC", "proba_convert"],
                                                target_variable="log_CPA",
                                                features_to_embed="industry"
                                                )
    print("############   Preprocessing for CPA prediction ###########")
    preproc_cpa_train_datastore = cpa_preproc_pipeline.run_preprocess_pipeline(categorical_target=None,
                                                                               encoder_type="target_encoding",
                                                                               save_dir="metadata_store_cpa",
                                                                               stats_to_compute="median",
                                                                               cal_sample_weights=False
                                                                              )
   
    
    preproc_cpa_train_target = preproc_cpa_train_datastore.target
    preproc_cpa_train_predictors = preproc_cpa_train_datastore.predictors
    preproc_cpa_train_predictor_colnames_inorder = preproc_cpa_train_datastore.predictor_colnames_inorder

    
    ### use the encoders created for train dataset to encode test dataset 
    # and prepare it for model evaluation. This prevents data leakage and 
    # ensures model evaluation reflect model performance in terms of the encoding 
    # used during training
    X_test_cpa_encoded = cpa_preproc_pipeline.encode_features(data=test_cpa,
                                                                stats_to_compute="median",
                                                                encoder_type="target_encoding"
                                                                )
    X_test_cpa_encoded_embed = cpa_preproc_pipeline.transform_columns_to_embed(data=X_test_cpa_encoded)
    preproc_cpa_test_datastore = cpa_preproc_pipeline.prepare_modelling_data(predictors=cpa_preproc_pipeline.predictors,
                                                                            embedding_colname=cpa_preproc_pipeline.embedding_colname,
                                                                            data=X_test_cpa_encoded_embed,
                                                                            target=cpa_preproc_pipeline.target_variable
                                                                            )
    preproc_cpa_test_predictors = preproc_cpa_test_datastore.predictors
    preproc_cpa_test_target = preproc_cpa_test_datastore.target
    preproc_cpa_test_predictor_colnames_inorder = preproc_cpa_test_datastore.predictor_colnames_inorder
    

    
    print("############# TRAINING CPA model  ##########")
    cpa_trainer = ModelTrainer(training_predictors=preproc_cpa_train_predictors,
                                training_target=preproc_cpa_train_target,
                                testing_predictors=preproc_cpa_test_predictors,
                                testing_target=preproc_cpa_test_target,
                                sample_weight=None,
                                model_type="KNeighborsRegressor"
                                )
    cpa_trainer.run_model_training_pipeline(cv=20, scoring=['neg_root_mean_squared_error'],
                                            model_result_metrics=['test_neg_root_mean_squared_error','train_neg_root_mean_squared_error'],
                                            evaluation_metric="root_mean_squared_error",
                                            save_model_as="cpa_regressor.model",
                                            save_dir="model_store"
                                            )
    cpa_regressor = cpa_trainer.estimator
    
    test_log_cpa_pred = cpa_regressor.predict(preproc_cpa_test_predictors).tolist()
    train_log_cpa_pred = cpa_regressor.predict(preproc_cpa_train_predictors).tolist()
   
    #test_cpa_pred = np.expm1(test_log_cpa_pred)
    #train_cpa_pred = np.expm1(train_log_cpa_pred)
    #print(f"CV Results: {cpa_trainer.cv_metrics}")
    
