from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import load_and_clean_data, build_xgboost_features
import numpy as np

app = FastAPI()

# ── Add this CORS block ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allows all origins (fine for local dev)
    allow_methods=["*"],
    allow_headers=["*"],
)
# ────────────────────────────────────────────────────

# Load once on startup
df = load_and_clean_data('demand_forecasting.csv')
df_feat = build_xgboost_features(df, target_col='Units_Sold')

@app.get("/api/forecast")
def get_forecast():
    # Use actual sales values as a stand-in for predictions until models are trained
    sample = df_feat.tail(30)  # last 30 rows as forecast window
    sales  = sample['Units_Sold'].tolist()
    dates  = sample['day_of_week'].tolist()

    # Simple confidence band using rolling std
    std = float(df_feat['Units_Sold'].std())
    return {
        "status": "success",
        "forecast_dates": dates,
        "predictions":  sales,
        "lower_bound":  [round(v - std, 2) for v in sales],
        "upper_bound":  [round(v + std, 2) for v in sales]
    }

@app.get("/api/anomalies")
def get_anomalies():
    # Flag rows where sales dropped below mean - 2*std
    mean = df_feat['Units_Sold'].mean()
    std  = df_feat['Units_Sold'].std()
    anomalies = df_feat[df_feat['Units_Sold'] < (mean - 2 * std)].tail(10)

    return {
        "anomalies_detected": [
            {
                "date":   row['Date'],
                "value":  row['Units_Sold'],
                "reason": "Breached lower bound (mean - 2σ)"
            }
            for _, row in anomalies.iterrows()
        ]
    }
