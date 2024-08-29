import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add indicators function
def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    return df

# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to create sequences
def create_sequences(data, seq_length, features):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, features.index('Adj Close')])
    return np.array(X), np.array(y)

# Function to generate recommendation
def generate_recommendation(price_change, mape, direction_accuracy):
    if price_change > 2 and direction_accuracy > 60:
        return "Strong Buy"
    elif price_change > 1 and direction_accuracy > 55:
        return "Buy"
    elif price_change < -2 and direction_accuracy > 60:
        return "Strong Sell"
    elif price_change < -1 and direction_accuracy > 55:
        return "Sell"
    else:
        return "Hold"

# Streamlit app
def main():
    st.set_page_config(page_title="LSTM Stock Price Predictor", page_icon="📈", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #1E90FF;
    }
    .stButton>button {
        color: #4CAF50;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    .stTextInput>div>div>input {
        color: #1E90FF;
    }
    .stProgress > div > div > div > div {
        background-color: #1E90FF;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">LSTM Stock Price Predictor 📈</p>', unsafe_allow_html=True)
    st.markdown("""
    This app uses an LSTM (Long Short-Term Memory) neural network to predict stock prices. 
    Upload a CSV file with historical stock data to get started!
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = add_indicators(df)
            
            features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_20', 'RSI', 'MACD', 'ATR']
            df.dropna(inplace=True)
            
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df[features])
            
            seq_length = 30
            X, y = create_sequences(data_scaled, seq_length, features)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_test = torch.FloatTensor(X_test).to(device)
            y_test = torch.FloatTensor(y_test).to(device)
            
            # Create DataLoader
            train_data = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            
            # Initialize and train model
            model = LSTMModel(len(features), 64, 2, 1).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f'Training Progress: {progress:.2%}')
            
            status_text.text('Training Complete!')
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                train_predictions = model(X_train).squeeze()
                test_predictions = model(X_test).squeeze()
                
                train_loss = criterion(train_predictions, y_train)
                test_loss = criterion(test_predictions, y_test)
            
            col1, col2 = st.columns(2)
            col1.metric("Train Loss", f"{train_loss.item():.4f}")
            col2.metric("Test Loss", f"{test_loss.item():.4f}")
            
            # Convert predictions back to original scale
            predictions_scaled = np.zeros((len(test_predictions), len(features)))
            predictions_scaled[:, features.index('Adj Close')] = test_predictions.cpu().numpy()
            predictions = scaler.inverse_transform(predictions_scaled)[:, features.index('Adj Close')]
            
            y_test_scaled = np.zeros((len(y_test), len(features)))
            y_test_scaled[:, features.index('Adj Close')] = y_test.cpu().numpy()
            y_test = scaler.inverse_transform(y_test_scaled)[:, features.index('Adj Close')]
            
            # Calculate metrics
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            price_direction_accuracy = np.mean((y_test[1:] > y_test[:-1]) == (predictions[1:] > predictions[:-1])) * 100
            
            col1, col2 = st.columns(2)
            col1.markdown(f'<p class="metric-value">Mean Absolute Percentage Error: {mape:.2f}%</p>', unsafe_allow_html=True)
            col2.markdown(f'<p class="metric-value">Price Direction Accuracy: {price_direction_accuracy:.2f}%</p>', unsafe_allow_html=True)
            
            # Plotting
            st.subheader("Stock Price Prediction")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.set_style("whitegrid")
            ax.plot(df['Date'].iloc[-len(y_test):], y_test, label='Actual', linewidth=2)
            ax.plot(df['Date'].iloc[-len(predictions):], predictions, label='Predicted', linewidth=2)
            ax.set_title('Stock Price Prediction', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Adjusted Close Price', fontsize=12)
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Next day prediction
            last_sequence = torch.FloatTensor(data_scaled[-seq_length:]).unsqueeze(0).to(device)
            next_day_pred_scaled = model(last_sequence).item()
            next_day_pred_scaled_full = np.zeros((1, len(features)))
            next_day_pred_scaled_full[0, features.index('Adj Close')] = next_day_pred_scaled
            next_day_pred = scaler.inverse_transform(next_day_pred_scaled_full)[0, features.index('Adj Close')]
            
            current_price = df['Adj Close'].iloc[-1]
            price_change = (next_day_pred - current_price) / current_price * 100
            
            recommendation = generate_recommendation(price_change, mape, price_direction_accuracy)
            
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<p class="metric-value">Current Price: ${current_price:.2f}</p>', unsafe_allow_html=True)
            col2.markdown(f'<p class="metric-value">Predicted Next Day Price: ${next_day_pred:.2f}</p>', unsafe_allow_html=True)
            col3.markdown(f'<p class="metric-value">Predicted Price Change: {price_change:.2f}%</p>', unsafe_allow_html=True)
            
            st.subheader("Recommendation")
            st.info(f"Based on the model's prediction: **{recommendation}**")
            
            st.warning("""
            **Note:** This recommendation is based on a machine learning model and should not be
            considered as financial advice. Always do your own research and consult with a
            financial advisor before making investment decisions.
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure you've uploaded a valid CSV file with the required columns: Date, Open, High, Low, Close, Adj Close")

if __name__ == '__main__':
    main()
