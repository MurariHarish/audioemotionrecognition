import os
import sys

import numpy as np 
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            logging.info("Started Fitting {} model",i)
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            grid_search=GridSearchCV(model,para,cv=3)
            grid_search.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_accuracy=accuracy_score(y_train,y_train_pred)
            test_accuracy=accuracy_score(y_test,y_test_pred)

            report[list(model.keys())[i]]=test_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)
