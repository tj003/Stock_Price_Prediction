# src/pipeline/predict_pipeline.py
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import DataIngestion
import logging
from dotenv import load_dotenv
load_dotenv()# to load API key

class PredictPipeline:
    def __init__(self):
        self.data_ingestor = DataIngestion()

    def predict(self, stock_symbol: str, task: str):
        try:
            df = self.data_ingestor.fetch_daily_stock_data(symbol=stock_symbol, api_key=os.getenv("ALPHAVANTAGE_API_KEY"))
            latest_data = df.tail(1)  # Use the latest data point for prediction

            if task == 'regression':
                model_path = 'artifact/model_reg.pkl'
                preprocessor_path = 'artifact/reg_preprocessor.pkl'
            else:
                model_path = 'artifact/model_cls.pkl'
                preprocessor_path = 'artifact/clf_preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            selected_features = ['open', 'high', 'low', 'close', 'volume', 'ma_5', 'ma_20', 'daily_return', 'rsi', 'macd', 'macd_signal']
            input_data = latest_data[selected_features]

            transformed_data = preprocessor.transform(input_data)
            prediction = model.predict(transformed_data)

            logging.info(f"Prediction using {task} model for {stock_symbol}: {prediction[0]}")
            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, stock_symbol: str, task: str):
        self.stock_symbol = stock_symbol
        self.task = task

    def get_inputs(self):
        return self.stock_symbol, self.task
