import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','raw_train_data.csv')
    test_data_path: str=os.path.join('artifacts','raw_test_data.csv')
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config =  DataIngestionConfig()

    def fetch_data(self,data):

        logging.info("Started Fetching data")
        crema_directory_list = os.listdir(data)
    
        audio_path = []
        audio_emotion = []
        for path in crema_directory_list:
            emotion = path.split('_')[2]
            audio_emotion.append(emotion)
            audio_path.append(os.path.join(data, path))

        audio_emotion_df = pd.DataFrame(audio_emotion,columns = ['target_emotion'])
        audio_emotion_df = audio_emotion_df.replace({'ANG':'angry', 'SAD':'sad','DIS':'disgust','HAP':'happy','FEA':'fear','NEU':'neutral'})

        audio_path = pd.DataFrame(audio_path, columns=['path'])

        raw_data = pd.concat([audio_path, audio_emotion_df], axis = 1)
        return raw_data
    
    def initiate_data_ingestion(self):

        logging.info("Initiating data ingestion config")
        try:
            data_path = 'notebook/data'
            raw_crema_df = self.fetch_data(data_path)
            logging.info("Data Fetch Completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            raw_crema_df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train Test split Initiated')

            raw_train_data,raw_test_data=train_test_split(raw_crema_df,test_size=0.2,random_state=42)
            raw_train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            raw_test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Successful')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    raw_train_data_path,raw_test_data_path = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    X_train,X_test,y_train,y_test = data_transformation.initiate_data_transformation(raw_train_data_path,raw_test_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train,X_test,y_train,y_test))