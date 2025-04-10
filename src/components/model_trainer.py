import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# define paths to save the model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('Artifact', "model_reg.pkl")
    trained_model_file_path_cls: str = os.path.join('Artifact', "model_cls.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, task: str = "regression"):
        try:
            logging.info("Starting model trainer pipeline.")
            logging.info("Splitting training and testing data arrays.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"📊 Task type detected: {task}")

            # ========= REGRESSION =========
            if task == "regression":
                logging.info("Initializing regression models and hyperparameters.")
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "AdaBoost Regressor": AdaBoostRegressor()
                }
                # Hyperparamters for regression models
                params = {
                    "Decision Tree": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                    "Random Forest Regressor": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Gradient Boosting Regressor": {
                        'learning_rate': [.1, .01, .05, .001],
                        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Linear Regression": {},
                    "XGBRegressor": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "AdaBoost Regressor": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }
                }

                model_path = self.model_trainer_config.trained_model_file_path

            # ========= CLASSIFICATION =========
            elif task == "classification":
                logging.info(" Initializing classification models and hyperparameters.")
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "XGBClassifier": XGBClassifier()
                }
                # Hyperparamters for classification models
                params = {
                    "Logistic Regression": {},
                    "Decision Tree Classifier": {
                        'criterion': ['gini', 'entropy', 'log_loss']
                    },
                    "Random Forest Classifier": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Gradient Boosting Classifier": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "AdaBoost Classifier": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "XGBClassifier": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }
                }

                model_path = self.model_trainer_config.trained_model_file_path_cls

            else:
                logging.error(f"Invalid task type '{task}' received.")
                raise ValueError(f"Invalid task type '{task}'. Must be 'regression' or 'classification'.")

            logging.info("Starting model evaluation...")
            model_report, trained_models = evaluate_model(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, param=params,
            task=task
)

            logging.info("Model evaluation completed.")

            # Filter out overfitted models (score == 1.0)
            filtered_model_report = {k: v for k, v in model_report.items() if v < 1.0}

            if not filtered_model_report:
                logging.warning("All models appear to be overfitting (score == 1.0).")
                raise CustomException("All models overfit — score == 1.0")

            best_model_score = max(filtered_model_report.values())
            best_model_name = max(filtered_model_report, key=filtered_model_report.get)
            best_model = trained_models[best_model_name]



            logging.info(f"Best model selected: {best_model_name} with score: {best_model_score}")

            if task == "regression" and best_model_score < 0.6:
                logging.warning(" No suitable regression model found (R² < 0.6)")
                raise CustomException("No best regression model found (R² < 0.6)")
            elif task == "classification" and best_model_score < 0.6:
                logging.warning(" No suitable classification model found (accuracy < 0.6)")
                raise CustomException("No best classification model found (accuracy < 0.6)")

            logging.info(f"Saving best {task} model: '{best_model_name}' to file: {model_path}")
            save_object(file_path=model_path, obj=best_model)
            logging.info(f"Model '{best_model_name}' has been successfully saved.")


            logging.info("Performing final evaluation on test set...")
            predictions = best_model.predict(X_test)

            final_score = r2_score(y_test, predictions) if task == "regression" else accuracy_score(y_test, predictions)
            logging.info(f"Final {task} score on test data: {final_score}")

            return final_score

        except Exception as e:
            logging.exception(" Exception occurred in initiate_model_trainer.")
            raise CustomException(e, sys)
