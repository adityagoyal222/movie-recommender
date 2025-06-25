import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.components.data_extraction import DataExtraction,DataExtractionConfig
from src.components.model_training import ModelTrainer,ModelTrainerConfig

class TrainPipeline:
    def __init__(self):
        pass

    def train_model(self):
        logging.info("Training Pipeline has started training")
        try:
            data_ingestion = DataIngestion()
            movies_path,ratings_path,_ = data_ingestion.initiate_data_ingestion()

            data_extraction = DataExtraction()
            train_data,test_data,_,_,_ = data_extraction.initiate_data_extraction(movies_path,ratings_path)

            model_trainer = ModelTrainer()
            report = model_trainer.initiate_model_training(train_data,test_data)

            return report

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.train_model()