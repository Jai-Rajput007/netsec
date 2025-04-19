import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging

load_dotenv()

MONGO_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            logging.info("Initializing NetworkDataExtract")
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise NetworkSecurityException(str(e), exc_traceback)

    def cv_to_json_converter(self, file_path):
        try:
            logging.info(f"Converting CSV to JSON: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise NetworkSecurityException(str(e), exc_traceback)

    def insert_data_mongodb(self, records, database, collection):
        try:
            logging.info(f"Inserting data into MongoDB: {database}.{collection}")
            self.mongo_client = pymongo.MongoClient(MONGO_URL, tlsCAFile=ca)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]
            
            result = self.collection.insert_many(records)
            return result.inserted_ids
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise NetworkSecurityException(str(e), exc_traceback)

if __name__ == '__main__':
    FILE_PATH = r"Network_Data\dataset_full.csv"
    DATABASE = "NETSEC"
    COLLECTION = "NetworkData"
    
    logging.info("Starting data extraction and insertion process")
    networkobj = NetworkDataExtract()
    
    try:
        records = networkobj.cv_to_json_converter(file_path=FILE_PATH)
        print(records)
        no_of_records = networkobj.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Number of records inserted: {len(no_of_records)}")
    except NetworkSecurityException as e:
        logging.error(f"Error: {e}")