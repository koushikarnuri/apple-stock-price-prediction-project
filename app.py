
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

# -----------------------
# FILES
# -----------------------
ARIMA_FILE = "arima_model.joblib"
SARIMA_FILE = "sarima_model.joblib"
RF_FILE = "random_forest_model.joblib"
XGB_FILE = "xgboost_model.joblib"
LSTM_FILE = "lstm_model.h5"
DATA_FILE = "AAPL.csv"

LOOKBACK = 60

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    return {
        "ARIMA": joblib.load(ARIMA_FILE),
        "SARIMA": joblib.load(SARIMA_FILE),
        "Random Forest": joblib.load(RF_FILE),
        "XGBoost": joblib.load(XGB_FILE),
        "LSTM": load_model(LSTM_FILE)
    }

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df.set_index("Date", inplace=True)

    df["MA07"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["Volatility"] = df["Close"].rolling(10).std()
    df["Daily_Returns"] = df["Close"].pct_change()

    df.dropna(inplace=True)

    return df

models = load_models()
df = load_data()

# -----------------------
# SCALER
# -----------------------
@st.cache_resource
def get_scaler():
    scaler = MinMaxScaler()
    scaler.fit(df[["Close"]])
    return scaler

scaler = get_scaler()

# -----------------------
# HEADER
# -----------------------
st.title("🍎 Apple Stock Price Prediction Dashboard")

# -----------------------
# PERFORMANCE TABLE
# -----------------------
performance_df = pd.DataFrame({
    "Model": ["ARIMA","SARIMA","Random Forest","XGBoost","LSTM"],
    "RMSE": [33.38,26.72,32.31,32.54,8.71],
    "MAPE": [0.1153,0.1052,0.0904,0.0915,0.0343]
})

st.subheader("Model Performance")

st.dataframe(performance_df)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("Prediction Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["ARIMA","SARIMA","Random Forest","XGBoost","LSTM"]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    90,
    30
)

# -----------------------
# FORECAST BUTTON
# -----------------------
if st.button("Generate Forecast"):

    forecast = None

    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    if selected_model == "ARIMA":

        res = models["ARIMA"].get_forecast(steps=forecast_days)
        forecast = pd.Series(
            res.predicted_mean,
            index=future_dates
        )

    elif selected_model == "SARIMA":

        res = models["SARIMA"].get_forecast(steps=forecast_days)
        forecast = pd.Series(
            res.predicted_mean,
            index=future_dates
        )

    elif selected_model in ["Random Forest","XGBoost"]:

        features = [
            "Open",
            "High",
            "Low",
            "Volume",
            "MA07",
            "MA30",
            "Volatility",
            "Daily_Returns"
        ]

        last_row = df[features].iloc[-1]

        future_df = pd.DataFrame(
            [last_row] * forecast_days,
            columns=features,
            index=future_dates
        )

        if selected_model == "Random Forest":
            preds = models["Random Forest"].predict(future_df)
        else:
            preds = models["XGBoost"].predict(future_df)

        forecast = pd.Series(preds, index=future_dates)

    elif selected_model == "LSTM":

        scaled = scaler.transform(df[["Close"]])

        batch = scaled[-LOOKBACK:]
        batch = batch.reshape((1, LOOKBACK, 1))

        preds = []

        for _ in range(forecast_days):

            pred = models["LSTM"].predict(
                batch,
                verbose=0
            )[0]

            preds.append(pred)

            batch = np.append(
                batch[:,1:,:],
                [[pred]],
                axis=1
            )

        preds = np.array(preds)

        forecast = scaler.inverse_transform(
            preds.reshape(-1,1)
        ).flatten()

        forecast = pd.Series(
            forecast,
            index=future_dates
        )

    st.success("Forecast Generated")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index[-100:],
            y=df["Close"].tail(100),
            mode="lines",
            name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            name="Forecast"
        )
    )

    fig.update_layout(
        title=f"{selected_model} Forecast",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    result_df = pd.DataFrame({
        "Date": forecast.index,
        "Predicted Price": forecast.values
    })

    st.subheader("Forecast Values")

    st.dataframe(result_df)

    st.download_button(
        "Download CSV",
        result_df.to_csv(index=False),
        "forecast.csv",
        "text/csv"
    )
```
