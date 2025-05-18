from netsec.components.data_ingestion import DataIngestion
from netsec.exception.exception import NetworkSecurityException
from netsec.logging.logger import logging
from netsec.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from netsec.entity.config_entity import TrainingPipelineConfig,ModelTrainerConfig
import sys
from netsec.components.data_validation import DataValidation
from netsec.components.data_transformation import DataTransformation
from netsec.components.model_trainer import ModelTrainer
from netsec.entity.artifact_entity import ModelTrainerArtifact

if __name__ == "__main__":
    try:
        logging.info("\n\n>>>>>> Pipeline started <<<<<<\n")
        
        # Initialize pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        
        # Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
        
        # Data Validation
        logging.info("Starting data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation completed: {data_validation_artifact}")
        
        # Data Transformation
        logging.info("Starting data transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation completed: {data_transformation_artifact}")
        
        # Model Training
        logging.info("Starting model training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, 
                                   data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model training completed: {model_trainer_artifact}")
        
        logging.info("\n\n>>>>>> Pipeline completed <<<<<<\n")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise NetworkSecurityException(e, sys)
