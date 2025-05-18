import os
import sys
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging

from netsec.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from netsec.entity.config_entity import ModelTrainerConfig

from netsec.utils.ml_utils.model.estimator import NetworkModel
from netsec.utils.main_utils.utils import save_object,load_object
from netsec.utils.main_utils.utils import load_numpy_array_data,evaluate_models

from netsec.utils.ml_utils.model.estimator import NetworkModel
from netsec.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'='*20} Model Trainer {'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def track_mlflow(sel,best_model,classificationmetric):
         with mlflow.start_run():
              f1_score = classificationmetric.f1_score
              precision_score = classificationmetric.precision_score
              recall_score = classificationmetric.recall_score 
              
              mlflow.log_metric("f1_score",f1_score)
              mlflow.log_metric("precision",precision_score)
              mlflow.log_metric("recall_score",recall_score)
              mlflow.sklearn.log_model(best_model,"model")


    def train_model(self,X_train,y_train,X_test,y_test):
            
            # Define models and their hyperparameters
            params = {
                "LogisticRegression": {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'n_jobs': [-1]  # Enable parallel processing
                },
                "RandomForestClassifier": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'n_jobs': [-1]  # Enable parallel processing
                },
                "DecisionTreeClassifier": {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy','log_loss']
                },
                "GradientBoostingClassifier": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'n_jobs': [-1]  # Enable parallel processing
                }
            }

            models = {
                "LogisticRegression": LogisticRegression(verbose=1, n_jobs=-1),  # Enable parallel
                "RandomForestClassifier": RandomForestClassifier(verbose=1, n_jobs=-1),  # Enable parallel
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1)  # Enable parallel
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,X_test=X_test,y_test=y_test,models=models,param=params)
              
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            y_train_pred =  best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            
            ##Track
            self.track_mlflow(best_model,classification_train_metric)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)
            self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=classification_train_metric,
                                 test_metric_artifact = classification_test_metric )
            
            return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing data")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Splitting input features and target")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            logging.info("Training the model")
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)