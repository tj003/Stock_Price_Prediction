import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill # helps to create an pickle file
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from src.logger import logging
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param, task):
    try:
        report = {}
        trained_models = {}

        for name, model in models.items():
            logging.info(f"⚙️ Training model: {name}")
            pipeline = GridSearchCV(model, param[name], cv=3, n_jobs=-1, verbose=1)
            pipeline.fit(X_train, y_train)
            logging.info(f"✅ Finished training: {name}")

            best_model = pipeline.best_estimator_

            if task == "regression":
                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)
            else:
                y_pred = best_model.predict(X_test)
                score = accuracy_score(y_test, y_pred)

            logging.info(f"📈 {name} score: {score}")
            report[name] = score
            trained_models[name] = best_model 

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)


    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)