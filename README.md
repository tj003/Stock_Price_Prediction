📈 IBM Stock Price Prediction App
A complete end-to-end machine learning pipeline for predicting next-day stock price movement, specifically built for IBM stock as a proof of concept.

This project was developed as part of a take-home assignment for ML Engineer - Condor AI, showcasing capabilities in data ingestion, feature engineering, model development, REST API creation, and containerization using Docker.

🚀 Features

- Fetches historical stock data using Financial Modeling Prep API.
- Engineers technical indicators like:
  - Moving Averages (5, 20-day)
  - Daily % change
  - Volume-related metrics
  - Technical Indicators using the `ta` library
- Predicts next-day stock direction (Up/Down) or regression (closing price).
- REST API built with Flask.
- Fully containerized with Docker.
- Clean, modular codebase following best practices.

⚙️ Setup Instructions
🔧 Local Environment

1. Clone the repository:
   git clone https://github.com/tj003/Stock_Price_Prediction.git
   cd Stock_Price_Prediction

2. Create virtual environment:
   python -m venv env

3. Activate the virtual environment:
   - On Linux/macOS: source env/bin/activate
   - On Windows: env\Scripts\activate

4. Install dependencies:
   pip install -r requirements.txt

5. Run the Flask app:
   python app.py

🐳 Docker Setup

1. Build the Docker image:
   docker build --no-cache -t ibm-stock-price-predictor -f Docker/Dockerfile .

2. Run the container:
   docker run -p 5000:5000 ibm-stock-price-predictor

API will be available at: http://localhost:5000

🔍 API Example

Sample Request (curl):
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{"stock_symbol":"IBM"}"

Sample Response:
{
  "predicted_movement": "UP",
  "predicted_price": 188.25
}

🤖 Model & Feature Engineering

Model: XGBoost Classifier or Regressor (configurable)
Feature Set Includes:
- 5-day & 20-day Moving Averages
- Momentum indicators
- Volume changes
- Daily returns
- Technical Indicators using `ta` library

💡 Assumptions

- Current version supports only IBM stock.
- Can be extended to other tickers by updating fetch logic and retraining.
- API keys are stored in a `.env` file.

🛠 Tech Stack

- Language: Python
- ML Libraries: scikit-learn, XGBoost
- Data: pandas, numpy, seaborn, matplotlib
- API: Flask
- Deployment: Docker

📬 Contact

Made with ❤️ by Tushar Jadhav
📧 Email: tusharj071@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/tushar-jadhav-1a3881225/

Let me know if you'd like:
- A version using FastAPI instead of Flask
- Graphs/logs of model performance added to README
- A `.env.example` template for easier setup

