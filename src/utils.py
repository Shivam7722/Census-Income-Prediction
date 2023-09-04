import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

from src.exception import CustomException
from src.logger import logging

import pickle 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Model Training
            model.fit(X_train, y_train)

            # Prediction on training data
            y_pred_training = model.predict(X_train)

            # Prediction on testing data
            y_pred_testing = model.predict(X_test)

            # Performance of models
            accuracy = accuracy_score(y_test,y_pred_testing)
            # f1_scores = f1_score(y_test,y_pred_testing)

            report[list(models.keys())[i]] =  accuracy

        return report
    
    except Exception as e:
        logging.info("Exception occured during the model training.")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)