from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging
import yaml
import os,sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path:str)-> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def load_object(file_path:str)-> None:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not exists")
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models, param) -> dict:
    try:
        report = {}
        
        # Scale features for all models
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, model in models.items():
            logging.info(f"\n{'='*20} Training {model_name} {'='*20}")
            para = param[model_name]
            
            # Adjust parameters for LogisticRegression
            if model_name == "LogisticRegression":
                model.set_params(max_iter=1000, tol=1e-4)
                if 'max_iter' not in para:
                    para['max_iter'] = [1000]
                if 'tol' not in para:
                    para['tol'] = [1e-4, 1e-3]
            
            # Configure GridSearchCV with parallel processing
            gs = GridSearchCV(
                model, para, 
                cv=3,
                n_jobs=-1,  # Enable parallel processing
                verbose=1,   # Show less verbose output
                scoring='f1'  # Use F1 score for classification
            )
            
            logging.info("Starting GridSearchCV fit")
            # Use scaled features for training
            gs.fit(X_train_scaled, y_train)
            
            logging.info(f"Best parameters found: {gs.best_params_}")
            
            # Get best parameters and fit model
            model.set_params(**gs.best_params_)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions using scaled features
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_f1': f1_score(y_train, y_train_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_f1': f1_score(y_test, y_test_pred),
                'test_precision': precision_score(y_test, y_test_pred),
                'test_recall': recall_score(y_test, y_test_pred)
            }
            
            logging.info(f"\nModel: {model_name} performance:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name}: {value:.4f}")
            
            # Use F1 score for model selection
            report[model_name] = metrics['test_f1']

        return report
    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise NetworkSecurityException(e, sys)
