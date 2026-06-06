import streamlit as st

st.set_page_config(
    page_title="Apple Stock Price Prediction Dashboard",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
# --- 1. Define Model and Data Paths ---
arima_filename = 'arima_model.joblib'
sarima_filename = 'sarima_model.joblib'
rf_filename = 'random_forest_model.joblib'
xgb_filename = 'xgboost_model.joblib'
lstm_model_filename = 'lstm_model.h5'
AAPL_DATA_FILE = 'AAPL.csv'

# --- 2. Load Models ---
@st.cache_resource
def load_all_models():
    """Loads all trained models."""
    try:
        arima_model = joblib.load(arima_filename)
        sarima_model = joblib.load(sarima_filename)
        rf_model = joblib.load(rf_filename)
        xgb_model = joblib.load(xgb_filename)
        lstm_model = load_model(lstm_model_filename)

        return {
            'ARIMA': arima_model,
            'SARIMA': sarima_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model,
            'LSTM': lstm_model,
        }
    except Exception as e:
        st.error(f"Error loading models. Please ensure all model files ({arima_filename}, {sarima_filename}, {rf_filename}, {xgb_filename}, {lstm_model_filename}) and {AAPL_DATA_FILE} are in the same directory. Error: {e}")
        st.stop()

models = load_all_models()

# --- 3. Load Original Data and Perform Feature Engineering ---
@st.cache_data
def load_and_preprocess_data():
    df_raw = pd.read_csv(AAPL_DATA_FILE)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)
    df_raw = df_raw.sort_values("Date")
    df_raw.set_index("Date", inplace=True)

    # Feature Engineering
    df_raw["MA07"] = df_raw["Close"].rolling(7).mean()
    df_raw["MA30"] = df_raw["Close"].rolling(30).mean()
    df_raw["Volatility"] = df_raw["Close"].rolling(10).std()
    df_raw["Daily_Returns"] = df_raw["Close"].pct_change()
    df_raw.dropna(inplace=True) # Drop NaNs introduced by rolling features

    return df_raw

df = load_and_preprocess_data()

# Constants from notebook
LOOKBACK = 60 # Lookback for LSTM model

# Re-initialize and fit scaler for LSTM within the app
@st.cache_resource
def get_lstm_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[['Close']].values)
    return scaler

lstm_scaler = get_lstm_scaler(df)

# --- Model Performance Data ---
model_performance_df = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Random Forest', 'XGBoost', 'LSTM'],
    'RMSE': [33.386831, 26.726207, 32.313498, 32.540103, 8.712613],
    'MAPE': [0.115383, 0.105267, 0.090483, 0.091569, 0.034375]
})

# Sort for better visualization and identifying the best model
model_performance_df = model_performance_df.sort_values(by='RMSE', ascending=True)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Apple Stock Price Prediction Dashboard")

# --- Model Performance Visualization ---
st.subheader("Model Performance Comparison")

# Combined RMSE and MAPE Bar Chart
fig = go.Figure()

# Add RMSE bars
fig.add_trace(go.Bar(
    x=model_performance_df['Model'],
    y=model_performance_df['RMSE'],
    name='RMSE',
    marker_color='blue'
))

# Add MAPE bars
fig.add_trace(go.Bar(
    x=model_performance_df['Model'],
    y=model_performance_df['MAPE'] * 100, # Multiply by 100 for percentage
    name='MAPE (%)',
    marker_color='darkblue'
))

fig.update_layout(
    barmode='group',
    title_text='Model Performance: RMSE and MAPE',
    xaxis_title='Model',
    yaxis_title='Metric Value',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.write("### All Model Metrics:")
st.dataframe(model_performance_df.round(4))

best_model = model_performance_df.iloc[0]
st.write(
    f"##### The best performing model (based on lowest RMSE and MAPE) is **{best_model['Model']}** "
    f"with RMSE: {best_model['RMSE']:.2f} and MAPE: {best_model['MAPE']:.4f}"
)

st.sidebar.header("Prediction Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model for Forecast:",
    ('ARIMA', 'SARIMA', 'Random Forest', 'XGBoost', 'LSTM')
)
prediction_horizon = st.sidebar.slider("Prediction Horizon (days):", 1, 90, 30)

st.write(f"### Using {selected_model_name} to predict the next {prediction_horizon} days of Apple Stock Prices")

# --- 5. Prediction Logic ---
forecast_prices = None
conf_int = None

# Create future dates based on the last date in the preprocessed DataFrame
future_dates = pd.date_range(
    start=df.index[-1] + timedelta(days=1),
    periods=prediction_horizon,
    freq="B" # Business days
)

# Get the last 100 historical prices to display with the forecast
historical_prices_for_plot = df["Close"].tail(100)

if st.button("Generate Forecast"):
    with st.spinner(f"Generating forecast using {selected_model_name}..."):
        if selected_model_name == 'ARIMA':
            forecast_res = models['ARIMA'].get_forecast(steps=prediction_horizon)
            forecast_prices = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

        elif selected_model_name == 'SARIMA':
            forecast_res = models['SARIMA'].get_forecast(steps=prediction_horizon)
            forecast_prices = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

        elif selected_model_name in ['Random Forest', 'XGBoost']:
            ml_features = ['Open', 'High', 'Low', 'Volume', 'MA07', 'MA30', 'Volatility', 'Daily_Returns']
            current_ml_df = df[ml_features].copy()

            # Repeat the last known feature set for simplicity over the prediction horizon
            last_known_features = current_ml_df.iloc[-1].to_dict()
            future_ml_features_df = pd.DataFrame([last_known_features] * prediction_horizon, index=future_dates)

            if selected_model_name == 'Random Forest':
                forecast_prices = models['Random Forest'].predict(future_ml_features_df)
            else: # XGBoost
                forecast_prices = models['XGBoost'].predict(future_ml_features_df)

            forecast_prices = pd.Series(forecast_prices, index=future_dates)
            conf_int = None 

        elif selected_model_name == 'LSTM':
            scaled_close_data = lstm
