import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
import requests
import ta
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ALPHAVANTAGE_API_KEY")

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
            
            df = self.add_ma_rsi_macd(df)
            df.dropna(inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info(f"[✔] Daily data for {symbol} saved to {self.ingestion_config.raw_data_path}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def add_ma_rsi_macd(self, df):
        

        # Reverse DataFrame to have oldest data first (required for indicators)
        df_rev = df[::-1].copy()

        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()

        # 📉 Daily % Change (Daily Return)
        df_rev["daily_return"] = df_rev["close"].pct_change()

        # 💹 RSI (Relative Strength Index)
        df_rev["rsi"] = ta.momentum.RSIIndicator(df_rev["close"], window=14).rsi()

        # 📊 MACD and Signal Line
        macd = ta.trend.MACD(df_rev["close"])
        df_rev["macd"] = macd.macd()
        df_rev["macd_signal"] = macd.macd_signal()

        # Reverse back to original format (newest row at top)
        df["daily_return"] = df_rev["daily_return"][::-1].values
        df["rsi"] = df_rev["rsi"][::-1].values
        df["macd"] = df_rev["macd"][::-1].values
        df["macd_signal"] = df_rev["macd_signal"][::-1].values
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df['next_day_close'] = df['close'].shift(-1)
        # Drop rows with NaNs caused by indicators or shift
        df.dropna(inplace=True)
        logging.info(f"MA, MACD, RSI columns added in dataset\n{df.head()}")


        return df


    def initiate_data_ingestion(self, symbol: str, api_key: str):
        
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
    train_data, test_data = obj.initiate_data_ingestion("IBM", api_key)

    # ===== REGRESSION PIPELINE =====

    regression_transformer  = DataTransformation()
    train_arr_reg, test_arr_reg, _ = regression_transformer.initiate_data_transformation(train_data, test_data, task="regression")
    regression_trainer = ModelTrainer()
    print("✅ Training Regression Model")
    regression_result = regression_trainer.initiate_model_trainer(
        train_arr_reg, test_arr_reg, task="regression"
    )
    print("📦 Regression Result:", regression_result)

    # ===== CLASSIFICATION PIPELINE =====
    classification_transformer = DataTransformation()
    train_arr_cls, test_arr_cls, _ = classification_transformer.initiate_data_transformation(
        train_data, test_data, task="classification"
    )

    classification_trainer = ModelTrainer()
    print("✅ Training Classification Model")
    classification_result = classification_trainer.initiate_model_trainer(
        train_arr_cls, test_arr_cls, task="classification"
    )
    print("📦 Classification Result:", classification_result)