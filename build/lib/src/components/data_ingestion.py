import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import yaml
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer




@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def fetch_daily_stock_data(self, symbol: str, api_key: str) -> pd.DataFrame:
        """
        Fetches daily stock data from Alpha Vantage and saves to raw_data_path.
        """
        try:
            url = (
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                f"&symbol={symbol}&outputsize=full&apikey={api_key}"
            )

            response = requests.get(url)
            data = response.json()

            if "Time Series (Daily)" not in data:
                raise CustomException("Failed to retrieve daily data from API", sys)

            daily_data = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(daily_data, orient='index')

            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info(f"[✔] Daily data for {symbol} saved to {self.ingestion_config.raw_data_path}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, symbol: str, api_key: str):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

            api_key = config["api_key"]
        logging.info("Entered the data ingestion method/component")

        try:
            df = self.fetch_daily_stock_data(symbol, api_key)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=True)

            logging.info("Ingestion of the data is completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
#  
if __name__ =="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data, test_data)

    # modelTrainer= ModelTrainer()
    # print(modelTrainer.initiate_model_trainer(train_arr, test_arr))
