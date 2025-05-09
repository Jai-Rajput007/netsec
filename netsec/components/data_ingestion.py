from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging

##Config file
from netsec.entity.config_entity import DataIngestionConfig
from netsec.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        try:
            logging.info(f"Connecting to MongoDB: {MONGO_DB_URL}")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Test connection first
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.mongo_client.server_info()  # This will raise an exception if connection fails
            
            logging.info(f"Connected to MongoDB. Accessing database: {database_name}")
            collection = self.mongo_client[database_name][collection_name]
                
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"DataFrame created with shape: {df.shape}")
            return df

        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise NetworkSecurityException(f"MongoDB Connection Error: Could not connect to server. Please check your connection string and ensure MongoDB is running. Error: {str(e)}", sys)
        except Exception as e:
            raise NetworkSecurityException(f"Error in export_collection_as_dataframe: {str(e)}", sys)
        finally:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index = False, header= True)
            return dataframe  # Added return statement
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed Split")
            logging.info("Excited the fn of split")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Exporting")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,index = False,header = True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,index = False,header = True
            )
            logging.info(f"Exported train and test file path")
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            dataframe = self.export_collection_as_dataframe()
            
            if dataframe is None or dataframe.empty:
                raise NetworkSecurityException(
                    "Failed to fetch data or data is empty from MongoDB", 
                    sys
                )
            
            logging.info(f"DataFrame shape: {dataframe.shape}")
            logging.info("Exporting data to feature store...")
            dataframe = self.export_data_into_feature_store(dataframe)
            
            if dataframe.shape[0] == 0:
                raise NetworkSecurityException(
                    "DataFrame is empty after feature store export", 
                    sys
                )
            
            logging.info("Splitting data into train and test sets...")
            self.split_data_as_train_test(dataframe)
            
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion completed. Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            raise NetworkSecurityException(f"Error in data ingestion: {str(e)}", sys)
