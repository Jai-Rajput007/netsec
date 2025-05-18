import sys
import os,numpy
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from netsec.constants.training_pipeline import TARGET_COLUMN
from netsec.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from netsec.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from sklearn.impute import KNNImputer
from  netsec.entity.config_entity import DataTransformationConfig
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging

from netsec.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            logging.info(f"{'='*20} Data Transformation {'='*20}")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            
            # Validate paths immediately
            if data_validation_artifact.valid_train_file_path is None:
                raise ValueError("Valid train file path is None")
            if data_validation_artifact.valid_test_file_path is None:
                raise ValueError("Valid test file path is None")
                
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            if file_path is None:
                raise ValueError(f"File path is None")
            if not os.path.exists(file_path):
                raise ValueError(f"File does not exist: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls)-> Pipeline:
        try:
          imputer:KNNImputer =  KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
          processor:Pipeline = Pipeline([("imputer",imputer)])
          return processor
        except Exception as e :
            raise NetworkSecurityException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            
            valid_train_path = self.data_validation_artifact.valid_train_file_path
            valid_test_path = self.data_validation_artifact.valid_test_file_path
            
            # Double check paths
            if valid_train_path is None or valid_test_path is None:
                raise ValueError(f"Invalid file paths. Train: {valid_train_path}, Test: {valid_test_path}")
                
            if not os.path.exists(valid_train_path) or not os.path.exists(valid_test_path):
                raise ValueError(f"Files not found. Train: {valid_train_path}, Test: {valid_test_path}")
            
            logging.info(f"Loading training data from: {valid_train_path}")
            train_df = DataTransformation.read_data(valid_train_path)
            logging.info(f"Loading test data from: {valid_test_path}")
            test_df = DataTransformation.read_data(valid_test_path)

            logging.info(f"Splitting input and target features")
            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(f"Target column {TARGET_COLUMN} not found in training data")
                
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)
            
            logging.info("Getting preprocessor object")
            preprocessor = self.get_data_transformer_object()

            logging.info("Transforming input features")
            preprocessor_object = preprocessor.fit(input_feature_train_df) 
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info("Creating train and test arrays")
            train_arr = numpy.c_[transformed_input_train_feature, numpy.array(target_feature_train_df)]
            test_arr = numpy.c_[transformed_input_test_feature, numpy.array(target_feature_test_df)]

            logging.info("Saving transformed data")
            # Create directories
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            logging.info("Preparing transformation artifact")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise NetworkSecurityException(e, sys)