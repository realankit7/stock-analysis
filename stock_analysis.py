import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title('Stock Price Prediction Dashboard')
st.write("""
This app predicts stock prices using multiple machine learning models.
Select a stock symbol and date range to get started.
""")

# Sidebar inputs
st.sidebar.header('User Input Parameters')
symbol = st.sidebar.text_input('Stock Symbol (e.g. TCS.NS)', 'TCS.NS')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))
model_choice = st.sidebar.selectbox(
    'Select Model', 
    ['Prophet', 'ARIMA', 'SARIMA', 'LSTM', 'Model Comparison']
)

# Download data function
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, interval='1d')
        if data.empty:
            st.error("No data found for the given symbol and date range.")
            return None
            
        # Create technical indicators
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=30).std()
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
data = load_data(symbol, start_date, end_date)

if data is None:
    st.stop()

# Show raw data
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(data)

# Plot closing price
st.subheader('Closing Price with Moving Averages')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Close'], label='Close Price', alpha=0.7)
ax.plot(data['MA_50'], label='50-day MA', linestyle='--')
ax.plot(data['MA_200'], label='200-day MA', linestyle='--')
ax.set_title(f'{symbol} Stock Price with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Model functions
def run_prophet(data):
    try:
        st.subheader('Prophet Model Forecast')
        prophet_data = data.reset_index()[['Date', 'Close', 'MA_50', 'MA_200', 'Volatility']]
        prophet_data.columns = ['ds', 'y', 'MA_50', 'MA_200', 'Volatility']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        model.add_regressor('MA_50')
        model.add_regressor('MA_200')
        model.add_regressor('Volatility')
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=90)
        future = future.merge(prophet_data[['ds', 'MA_50', 'MA_200', 'Volatility']], on='ds', how='left')
        future = future.fillna(method='ffill').fillna(method='bfill')
        
        forecast = model.predict(future)
        
        # Plot forecast
        fig = model.plot(forecast)
        plt.title(f'{symbol} Stock Price Forecast with Prophet')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(fig)
        
        # Evaluation
        train_data = prophet_data[:-90]
        test_data = prophet_data[-90:]
        
        model_eval = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        model_eval.add_regressor('MA_50')
        model_eval.add_regressor('MA_200')
        model_eval.add_regressor('Volatility')
        model_eval.fit(train_data)
        
        future_test = pd.DataFrame({'ds': test_data['ds']})
        future_test = future_test.merge(test_data[['ds', 'MA_50', 'MA_200', 'Volatility']], on='ds', how='left')
        future_test = future_test.ffill().bfill()
        
        forecast_test = model_eval.predict(future_test)
        actual_values = test_data['y'].values
        predicted_values = forecast_test['yhat'].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        r2 = r2_score(actual_values, predicted_values)
        accuracy = 100 * (1 - (mae / actual_values.mean()))
        
        st.subheader('Prophet Model Evaluation')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("Accuracy", f"{accuracy:.2f}%")
        
        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data['ds'], actual_values, label='Actual Price', alpha=0.7)
        ax.plot(test_data['ds'], predicted_values, label='Predicted Price', linestyle='--')
        ax.set_title('Prophet Forecast vs Actual Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
    except Exception as e:
        st.error(f"Error in Prophet model: {str(e)}")
        return None

def run_arima(data):
    try:
        st.subheader('ARIMA Model Forecast')
        arima_data = data['Close'].reset_index(drop=True)
        
        # Split data
        train_size = int(len(arima_data) * 0.8)
        train_data_initial = arima_data[:train_size]
        test_data = arima_data[train_size:]
        
        # Rolling forecast
        predictions = []
        for i in range(len(test_data)):
            current_train_data = arima_data[:train_size + i]
            model = ARIMA(current_train_data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1).iloc[0]
            predictions.append(forecast)
        
        predictions_series = pd.Series(predictions, index=arima_data.index[train_size:])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(arima_data.index[:train_size], train_data_initial, label='Training Data')
        ax.plot(arima_data.index[train_size:], test_data, label='Testing Data')
        ax.plot(arima_data.index[train_size:], predictions_series, label='Predictions')
        ax.set_title(f'{symbol} Stock Price Prediction with ARIMA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Evaluation
        mae = mean_absolute_error(test_data, predictions_series)
        rmse = np.sqrt(mean_squared_error(test_data, predictions_series))
        r2 = r2_score(test_data, predictions_series)
        accuracy = 100 * (1 - (mae / test_data.mean()))
        
        st.subheader('ARIMA Model Evaluation')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("Accuracy", f"{accuracy:.2f}%")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
    except Exception as e:
        st.error(f"Error in ARIMA model: {str(e)}")
        return None

def run_sarima(data):
    try:
        st.subheader('SARIMA Model Forecast')
        arima_data = data['Close'].reset_index(drop=True)
        
        # Split data
        train_size = int(len(arima_data) * 0.8)
        train = arima_data[:train_size]
        test_data = arima_data[train_size:]
        
        # Fit SARIMA model
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        
        # Forecast
        forecast = results.get_forecast(steps=len(test_data))
        predicted_values = forecast.predicted_mean
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train.index[-100:], train[-100:], label='Training Data')
        ax.plot(test_data.index, test_data, label='Actual Price')
        ax.plot(test_data.index, predicted_values, label='SARIMA Predictions')
        ax.set_title(f'{symbol} Stock Price Prediction with SARIMA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
        
        # Evaluation
        mae = mean_absolute_error(test_data, predicted_values)
        rmse = np.sqrt(mean_squared_error(test_data, predicted_values))
        r2 = r2_score(test_data, predicted_values)
        accuracy = 100 * (1 - (mae / test_data.mean()))
        
        st.subheader('SARIMA Model Evaluation')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("Accuracy", f"{accuracy:.2f}%")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
    except Exception as e:
        st.error(f"Error in SARIMA model: {str(e)}")
        return None

def run_lstm(data):
    try:
        st.subheader('LSTM Model Forecast')
        st.warning("LSTM training may take several minutes. Please be patient.")
        
        # Prepare features
        features = data[['Close', 'MA_50', 'MA_200', 'Volatility']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data)-seq_length-1):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)
        
        seq_length = 60
        X, y = create_sequences(scaled_features, seq_length)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Plot training history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Make predictions
        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(
            np.concatenate([
                predicted,
                np.zeros((len(predicted), features.shape[1]-1))
            ], axis=1)
        )[:, 0]
        
        actual_prices = scaler.inverse_transform(
            np.concatenate([
                y_test.reshape(-1, 1),
                np.zeros((len(y_test), features.shape[1]-1))
            ], axis=1)
        )[:, 0]
        
        # Evaluation
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        r2 = r2_score(actual_prices, predicted_prices)
        accuracy = 100 * (1 - (mae / actual_prices.mean()))
        
        st.subheader('LSTM Model Evaluation')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("Accuracy", f"{accuracy:.2f}%")
        
        # Plot predictions
        lstm_plot_index = data.index[train_size + seq_length + 1:train_size + seq_length + 1 + len(y_test)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lstm_plot_index, actual_prices, label='Actual Price')
        ax.plot(lstm_plot_index, predicted_prices, label='Predicted Price')
        ax.set_title(f'{symbol} Stock Price Prediction with LSTM')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        }
        
    except Exception as e:
        st.error(f"Error in LSTM model: {str(e)}")
        return None

def compare_models(metrics):
    st.subheader('Model Performance Comparison')
    
    if not metrics:
        st.warning("No model metrics available for comparison")
        return
    
    # Create metrics DataFrame
    df = pd.DataFrame(metrics).T
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df.index))
    width = 0.2
    
    for i, metric in enumerate(['MAE', 'RMSE', 'Accuracy']):
        ax.bar(x + i*width, df[metric], width, label=metric)
    
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.75)
    st.pyplot(fig)
    
    # Determine best model
    best_rmse = df['RMSE'].idxmin()
    best_accuracy = df['Accuracy'].idxmax()
    
    st.success(f"Best model by RMSE: **{best_rmse}** (RMSE: {df.loc[best_rmse, 'RMSE']:.4f})")
    st.success(f"Best model by Accuracy: **{best_accuracy}** (Accuracy: {df.loc[best_accuracy, 'Accuracy']:.2f}%)")

# Run selected model
metrics = {}

if model_choice == 'Prophet':
    prophet_metrics = run_prophet(data)
    if prophet_metrics:
        metrics['Prophet'] = prophet_metrics

elif model_choice == 'ARIMA':
    arima_metrics = run_arima(data)
    if arima_metrics:
        metrics['ARIMA'] = arima_metrics

elif model_choice == 'SARIMA':
    sarima_metrics = run_sarima(data)
    if sarima_metrics:
        metrics['SARIMA'] = sarima_metrics

elif model_choice == 'LSTM':
    lstm_metrics = run_lstm(data)
    if lstm_metrics:
        metrics['LSTM'] = lstm_metrics

elif model_choice == 'Model Comparison':
    st.warning("This will run all models and may take significant time")
    
    if st.button('Run Full Comparison'):
        with st.spinner('Running Prophet...'):
            prophet_metrics = run_prophet(data)
            if prophet_metrics:
                metrics['Prophet'] = prophet_metrics
        
        with st.spinner('Running ARIMA...'):
            arima_metrics = run_arima(data)
            if arima_metrics:
                metrics['ARIMA'] = arima_metrics
        
        with st.spinner('Running SARIMA...'):
            sarima_metrics = run_sarima(data)
            if sarima_metrics:
                metrics['SARIMA'] = sarima_metrics
        
        with st.spinner('Running LSTM...'):
            lstm_metrics = run_lstm(data)
            if lstm_metrics:
                metrics['LSTM'] = lstm_metrics
        
        compare_models(metrics)

# Add footer
st.markdown("---")
st.caption("Stock Prediction App | Created with Streamlit")