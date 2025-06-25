import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts','model.pkl')
    embeddings_path: str = os.path.join('artifacts','embeddings.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_data,test_data):
        logging.info("Model Training has started!")
        try:
            model_path = self.model_trainer_config.model_path
            embeddings_path = self.model_trainer_config.embeddings_path

            report = evaluate_model(model_path,embeddings_path,train_data,test_data)

            logging.info("Model has been trained!")

            return report

        except Exception as e:
            raise CustomException(e,sys)