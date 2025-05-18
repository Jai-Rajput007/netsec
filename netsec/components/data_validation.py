from netsec.entity.artifact_entity import DataIngestionArtifact
from netsec.entity.artifact_entity import DataValidationArtifact
from netsec.entity.config_entity import DataValidationConfig
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging
from netsec.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from netsec.utils.main_utils.utils import read_yaml_file, write_yaml_file
import pandas as pd
import os,sys

class DataValidation:  
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            logging.info(f"Loading schema from: {SCHEMA_FILE_PATH}")
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info(f"Schema loaded successfully. Config: {self._schema_config}")
        except Exception as e:
            logging.error(f"Error loading schema: {str(e)}")
            raise NetworkSecurityException(e,sys)


    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Extract column names from schema config
            schema_columns = []
            for col_dict in self._schema_config['columns']:
                schema_columns.extend(col_dict.keys())
            
            df_columns = dataframe.columns.tolist()

            if len(df_columns) != len(schema_columns):
                logging.error(f"Number of columns mismatch. Expected {len(schema_columns)}, got {len(df_columns)}")
                
            missing_cols = set(schema_columns) - set(df_columns)
            extra_cols = set(df_columns) - set(schema_columns)
            
            if missing_cols:
                logging.error(f"Missing columns in dataset: {missing_cols}")
            if extra_cols:
                logging.error(f"Extra columns in dataset: {extra_cols}")
                
            return len(missing_cols) == 0 and len(extra_cols) == 0
        except Exception as e:
            logging.error(f"Error in validate_number_of_columns: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df) -> bool:
        try:
            logging.info("Checking if dataset drift occurred")
            
            # Get schema columns
            schema_columns = []
            for col_dict in self._schema_config['columns']:
                schema_columns.extend(col_dict.keys())
            
            drift_report = {}
            drift_report["status"] = True
            drift_report["columns"] = {}
            
            if not all(col in current_df.columns for col in schema_columns):
                raise ValueError("Current dataframe missing columns from schema")

            if base_df.isnull().values.any() or current_df.isnull().values.any():
                logging.warning("Datasets contain missing values - handling them before drift detection")
                base_df = base_df.fillna(base_df.mean())
                current_df = current_df.fillna(current_df.mean())
            
            for column in schema_columns:
                base_data = base_df[column]
                current_data = current_df[column]
                
                # Handle categorical columns by converting to numerical
                if base_data.dtype == 'object':
                    logging.info(f"Converting categorical column {column} to numerical")
                    base_data = pd.Categorical(base_data).codes
                    current_data = pd.Categorical(current_data).codes
                
                # Convert to numpy arrays and reshape
                base_data = base_data.values.reshape(-1)
                current_data = current_data.values.reshape(-1)
                
                try:
                    # Perform KS test
                    statistic, p_value = ks_2samp(base_data, current_data)
                    
                    drift_report["columns"][column] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift_detected": bool(p_value < 0.05)
                    }
                    
                    # If p-value is less than threshold, drift is detected
                    if p_value < 0.05:  # You can adjust this threshold
                        drift_report["status"] = False
                        logging.info(f"Drift detected in column: {column}, p-value: {p_value:.4f}")
                except Exception as e:
                    logging.warning(f"Could not perform drift detection on column {column}: {str(e)}")
                    drift_report["columns"][column] = {
                        "error": str(e)
                    }
                    continue
            
            # Create directory for drift report if it doesn't exist
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            
            # Save the drift report
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report
            )
            
            logging.info(f"Drift report saved at: {self.data_validation_config.drift_report_file_path}")
            logging.info(f"Drift status: {'Drift detected' if not drift_report['status'] else 'No drift detected'}")
            
            return drift_report["status"]
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""
            logging.info("Starting data validation")
            
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Ensure the files exist
            if not os.path.exists(train_file_path):
                raise ValueError(f"Training file {train_file_path} does not exist")
            if not os.path.exists(test_file_path):
                raise ValueError(f"Test file {test_file_path} does not exist")

            logging.info(f"Reading training data from {train_file_path}")
            train_dataframe = DataValidation.read_data(train_file_path)
            
            logging.info(f"Reading test data from {test_file_path}")
            test_dataframe = DataValidation.read_data(test_file_path)

            # Create directories
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)

            logging.info("Validating number of columns")
            validation_status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not validation_status:
                error_message += "Train dataframe does not contain all columns. "
                logging.warning(error_message)
                # Save to invalid directory
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
            else:
                # Save to valid directory
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info("Checking for dataset drift")
            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status and drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path if validation_status else None,
                valid_test_file_path=self.data_validation_config.valid_test_file_path if validation_status else None,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path if not validation_status else None,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path if not validation_status else None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            
            if not validation_status:
                raise ValueError(error_message)
                
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)