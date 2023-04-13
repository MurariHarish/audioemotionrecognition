import os
import sys
from dataclasses import dataclass

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:
            models = {
                  'MLP Classifier': MLPClassifier(max_iter=500),
             }
            params={
                 'MLP Classifier':{
                        'hidden_layer_sizes': [(10,)],
                        'random_state':[42],
                 }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            
            # To get best model score from dict
            Best_Model_Accuracy = max(sorted(model_report.values()))

            # To get best model name from dict
            Best_Model_Name = list(model_report.keys())[
                list(model_report.values()).index(Best_Model_Accuracy)
            ]

            # if Best_Model_Accuracy<0.6:
            #     raise CustomException("No best model found")
            # else:
            #     logging.info(f"Best found model on both training and testing dataset")

            Best_Model = models[Best_Model_Name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=Best_Model
            )
            logging.info("Best model saved as pickle")
            logging.info("Predicting Accuracy for X_test")
            y_test_pred=Best_Model.predict(X_test)

            Model_Accuracy=accuracy_score(y_test_pred,y_test)
            return Best_Model_Name, Model_Accuracy
            
        except Exception as e:
                raise CustomException(e,sys)