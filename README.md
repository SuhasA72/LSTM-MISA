## LSTM-MISA (Long Short-Term Memory - Multi-Indicator Stock Analyzer) 📈


This repository contains a web application built using Streamlit that leverages an LSTM (Long Short-Term Memory) neural network to predict stock prices. The application allows users to upload historical stock data in CSV format, process the data, and make predictions about future stock prices.

## Features

    Data Upload: Upload a CSV file containing historical stock data.
    
    Data Preprocessing: Automatically calculate technical indicators like SMA, RSI, MACD, and ATR.
    
    Model Training: Train an LSTM model on the uploaded data.
    
    Prediction: Predict future stock prices and calculate the mean absolute percentage error (MAPE) and price direction accuracy.
    
    Visualization: Visualize the actual and predicted stock prices on an interactive plot.
    
    Recommendation: Get a recommendation (Buy, Sell, Hold) based on the model's prediction.

## Installation

  To install and run this project locally, follow these steps:
  Prerequisites

  Ensure you have Python 3.10 or higher installed. Clone the repository to your local machine:

      git clone https://github.com/SuhasA72/LSTM-MISA.git
      cd LSTM-MISA
  
## Install Dependencies

  #You can install the required dependencies using pip:

      pip install -r requirements.txt

## Usage

## Run the Streamlit app

    streamlit run app.py

## Upload a CSV file containing the stock data. The file should have the following columns:

    Date
    Open
    High
    Low
    Close
    Adj Close

## Contributing

  If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the [MIT License](LICENSE.md). See the `LICENSE.md` file for details.


## Disclaimer

  Note: This application provides stock price predictions based on a machine learning model and should not be considered as financial advice. Always conduct your own research and consult with a financial advisor before making any investment decisions.


