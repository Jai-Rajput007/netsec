from netsec.components.data_ingestion import DataIngestion
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging
from netsec.entity.config_entity import DataIngestionConfig,DataValidationConfig
from netsec.entity.config_entity import TrainingPipelineConfig
import sys
from netsec.components.data_validation import DataValidation

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
    except Exception as e:
        raise NetworkSecurityException(e, sys)
