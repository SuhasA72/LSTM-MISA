
# LSTM-MISA (Long Short-Term Memory - Multi-Indicator Stock Analyzer) 📈

Welcome to **LSTM-MISA**, a powerful web application designed to assist with stock price prediction using advanced machine learning techniques. This tool utilizes an LSTM (Long Short-Term Memory) neural network, a type of recurrent neural network (RNN) well-suited for time series forecasting, to analyze and predict stock prices based on historical data. The app is built with Streamlit, providing an intuitive and interactive user interface.

## 🔥 Features

- **Data Upload**: Easily upload a CSV file containing historical stock data, ensuring a seamless start to your analysis. The file should include essential columns like Date, Open, High, Low, Close, and Adjusted Close.

- **Data Preprocessing**: The app automatically processes the uploaded data by calculating key technical indicators such as:
  - **SMA**: Simple Moving Average, which smooths out price data to identify trends.
  - **RSI**: Relative Strength Index, a momentum oscillator that measures the speed and change of price movements.
  - **MACD**: Moving Average Convergence Divergence, a trend-following momentum indicator.
  - **ATR**: Average True Range, which measures market volatility.

- **Model Training**: Train an LSTM model tailored to your dataset. The model learns from historical data patterns to make informed predictions about future stock prices.

- **Prediction**: Generate future stock price predictions, accompanied by key metrics such as Mean Absolute Percentage Error (MAPE) and price direction accuracy, to evaluate the model's performance.

- **Visualization**: Interactive plots allow you to visualize actual vs. predicted stock prices, providing insights into the model's accuracy and trends.

- **Recommendation**: Based on the model's predictions, receive actionable recommendations on whether to Buy, Sell, or Hold the stock.

## 📦 Installation

To get started with **LSTM-MISA**, follow the instructions below to set up the project on your local machine.

### Prerequisites

Ensure you have Python 3.10 or higher installed on your system. Then, clone the repository:

```bash
git clone https://github.com/SuhasA72/LSTM-MISA.git
cd LSTM-MISA
```

### Install Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Streamlit App

Launch the application by running the following command:

```bash
streamlit run app.py
```

### Uploading Data

- Upload a CSV file containing the stock data you wish to analyze.
- The CSV file should include the following columns:
  - **Date**: The date of the stock data entry.
  - **Open**: The opening price of the stock on that date.
  - **High**: The highest price of the stock on that date.
  - **Low**: The lowest price of the stock on that date.
  - **Close**: The closing price of the stock on that date.
  - **Adj Close**: The adjusted closing price, accounting for dividends and splits.

After uploading, the app will preprocess the data, train the LSTM model, and provide predictions and visualizations based on your data.


## 📄 License

This project is licensed under the [MIT License](LICENSE.md). For more details, see the `LICENSE.md` file.

## ⚠️ Disclaimer

**LSTM-MISA** provides stock price predictions based on a machine learning model. These predictions are for educational purposes and should not be construed as financial advice. Please conduct your own research and consult with a financial advisor before making any investment decisions.

---

