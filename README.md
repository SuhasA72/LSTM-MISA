# LSTM-MISA (Long Short-Term Memory - Multi-Indicator Stock Analyzer) 📈 | Predict Stock Prices Using Machine Learning

[![GitHub license](https://img.shields.io/github/license/SuhasA72/LSTM-MISA.svg)](https://github.com/SuhasA72/LSTM-MISA/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/release/SuhasA72/LSTM-MISA.svg)](https://GitHub.com/SuhasA72/LSTM-MISA/releases/)

Welcome to **LSTM-MISA**, a powerful web application designed to assist with stock price prediction using advanced machine learning techniques. This tool utilizes an LSTM (Long Short-Term Memory) neural network, a type of recurrent neural network (RNN) well-suited for time series forecasting, to analyze and predict stock prices based on historical data. The app is built with Streamlit, providing an intuitive and interactive user interface.

## Table of Contents

- [Why LSTM for Stock Prediction?](#why-lstm-for-stock-prediction)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Citation](#citation)
- [Roadmap](#roadmap)
- [Support](#support)
- [License](#-license)
- [Disclaimer](#️-disclaimer)

## Why LSTM for Stock Prediction?

LSTM networks are particularly effective for stock price prediction due to their ability to capture long-term dependencies in time series data. Unlike traditional RNNs, LSTMs can remember or forget information over long periods, making them ideal for identifying patterns in stock market trends that may span days, weeks, or even months.

## 🔥 Features

- **Data Upload**: Easily upload a CSV file containing historical stock data.
- **Data Preprocessing**: Automatic calculation of key technical indicators (SMA, RSI, MACD, ATR).
- **Model Training**: Train an LSTM model tailored to your dataset.
- **Prediction**: Generate future stock price predictions with performance metrics.
- **Visualization**: Interactive plots for actual vs. predicted stock prices.
- **Recommendation**: Actionable Buy, Sell, or Hold recommendations.

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/SuhasA72/LSTM-MISA.git
cd LSTM-MISA
```

### Set Up a Virtual Environment

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

### Uploading Data

1. Prepare a CSV file with columns: Date, Open, High, Low, Close, Adj Close.
2. Upload the CSV file through the web interface.
3. You can download the csv files from the yahoofinance or google and upload it.
4. The app will preprocess the data, train the LSTM model, and provide predictions and visualizations.

## Citation

If you use LSTM-MISA in your research, please cite it as follows:

```bibtex
@software{lstm_misa,
  author = {SuhasA72},
  title = {LSTM-MISA: Long Short-Term Memory Multi-Indicator Stock Analyzer},
  year = {2024},
  url = {https://github.com/SuhasA72/LSTM-MISA}
}
```

## Support

If you need help or have any questions, please [open an issue](https://github.com/SuhasA72/LSTM-MISA/issues) on the GitHub repository

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## ⚠️ Disclaimer

**LSTM-MISA** provides stock price predictions based on a machine learning model. These predictions are for educational purposes only and should not be construed as financial advice. Please conduct your own research and consult with a financial advisor before making any investment decisions.

---
