# Apple Stock Price Prediction — 30-Day Forecasting

🚀 Live App: https://apple-stock-price-prediction-project-fwpa9gi6st5kc3zwu9wjmy.streamlit.app/

## Overview
A time-series forecasting system to predict Apple (AAPL) stock prices
30 days ahead using classical, machine learning, and deep learning models.
Trained on historical data from 2012–2019 using multiple algorithms and compared for best performance.

## Results
| Model         | Performance                      |
|---------------|----------------------------------|
| SARIMA        | Best — 8% lower RMSE vs baseline |
| LSTM          | 8% lower RMSE vs baseline        |
| XGBoost       | Strong short-term prediction     |
| Random Forest | Solid baseline comparison        |
| ARIMA         | Classical baseline model         |

## Dataset
- **File:** `AAPL.xlsx`
- **Source:** Yahoo Finance / Kaggle
- **Period:** 2012 – 2019
- **Ticker:** AAPL (Apple Inc.)

## Models Used
- **ARIMA / SARIMA** — Classical time-series forecasting → `arima_model.joblib`, `sarima_model.joblib`
- **XGBoost** — Gradient boosting for regression → `xgboost_model.joblib`
- **Random Forest** — Ensemble learning baseline → `random_forest_model.joblib`
- **LSTM** — Deep learning sequence model → `lstm_model.h5`

## Tech Stack
Python | pandas | NumPy | scikit-learn | XGBoost |
TensorFlow | Keras | Statsmodels | Matplotlib | Plotly

## Project Structure
```
apple-stock-price-prediction-project/
│
├── time_series_forcasting.ipynb   ← Main Jupyter notebook
├── app.py                          ← Application/deployment script
├── requirements.txt                ← Required libraries
├── AAPL.xlsx                       ← Stock price dataset (2012–2019)
├── apple.pptx                      ← Project presentation
├── arima_model.joblib              ← Saved ARIMA model
├── sarima_model.joblib             ← Saved SARIMA model
├── random_forest_model.joblib      ← Saved Random Forest model
├── xgboost_model.joblib            ← Saved XGBoost model
└── lstm_model.h5                   ← Saved LSTM deep learning model
```

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Open the main notebook**
```bash
jupyter notebook time_series_forcasting.ipynb
```

**3. Or run the app**
```bash
python app.py
```

## Key Findings
- SARIMA and LSTM outperformed tree-based models for long-range 30-day forecasting
- Seasonal patterns identified in Apple stock with strong Q4 trends
- All trained models saved as `.joblib` and `.h5` files for reuse and deployment
- Ensemble approach reduced forecasting error significantly vs single-model baseline

## Author
**Chenna Sai Mani Koushik Arnuri**
- Email: koushikarnuri@gmail.com
- LinkedIn: [linkedin.com/in/koushik-arnuri-920195387](https://www.linkedin.com/in/koushik-arnuri-920195387)
- GitHub: [github.com/koushikarnuri](https://github.com/koushikarnuri)
