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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model_instance in models.items():  # Update the loop
            logging.info("Started Fitting %s model", model_name)

            para = param[model_name]

            grid_search = GridSearchCV(model_instance, para, cv=3)
            grid_search.fit(X_train, y_train)

            model_instance.set_params(**grid_search.best_params_)
            model_instance.fit(X_train, y_train)

            y_train_pred = model_instance.predict(X_train)
            y_test_pred = model_instance.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_accuracy  # Update the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
