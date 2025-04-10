import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #to apply all data encoing method in piplines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
# define paths to save the transformed data
@dataclass
class DataTransformationConfig:
    reg_preprocessor_path: str = os.path.join('artifact', 'reg_preprocessor.pkl')
    clf_preprocessor_path: str = os.path.join('artifact', 'clf_preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformaton_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsile for data transformation        
        '''
        try:
            numerical_columns = ['open', 'high', 'low', 'close', 'volume', 'ma_5', 'ma_20', 'daily_return', 'rsi', 'macd', 'macd_signal']
           
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical columns :{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path, task: str = "regression"):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Set target column and preprocessor path based on task
            if task == "regression":
                target_column_name = "next_day_close"
                preprocessor_path = self.data_transformaton_config.reg_preprocessor_path
            elif task == "classification":
                target_column_name = "target"
                preprocessor_path = self.data_transformaton_config.clf_preprocessor_path
            else:
                raise ValueError("Invalid task type. Use 'regression' or 'classification'")

            logging.info(f"Obtaining preprocessor object for task: {task}")
            preprocessor_obj = self.get_data_transformer_object()
            
            # Split into input and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training and testing data")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing object to: {preprocessor_path}")

            save_object(
                file_path=preprocessor_path,
                obj=preprocessor_obj
            )

            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)


        
            