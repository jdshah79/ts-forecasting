from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import load_and_clean_data, build_xgboost_features
import numpy as np

app = FastAPI(title="Dynamic Enterprise Forecasting Engine")

# ── Add this CORS block ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allows all origins (fine for local dev)
    allow_methods=["*"],
    allow_headers=["*"],
)
# ────────────────────────────────────────────────────

@app.get("/api/forecast")
def get_forecast(region: str = Query("All", description="Filter by a specific Region"), days_history: int = Query(90, desciption="Number of data points to be displayed"),
                 confidence_multiplier: float = Query(1.5, description="Multiplier for the historic error standard deviation to widen/narrow bounds"),
                 seasonality_mode: str = Query("additive", description="Seasonality type: 'additive' or 'multiplicative'")
                 ):
    # Load once on startup
    df = load_and_clean_data('demand_forecasting.csv')
    df_feat = build_xgboost_features(df, target_col='Units_Sold')
    
    # 2. DYNAMIC REGION FILTERING
    if region != "All" and "Region" in df.columns:
        df = df[df['Region'] == region]

    # 3. DYNAMIC HISTORY LENGTH
    # Take the last N rows/days requested by the user
    df = df.tail(days_history).reset_index(drop=True)

    # 4. CALCULATION OF HISTORIC ERROR (Your Confidence Band Logic)
    # Calculate the variation between what was ordered vs true demand historically
    if 'Units_Ordered' in df.columns and 'Demand' in df.columns:
        historic_errors = (df['Units_Ordered'] - df['Demand']).abs()
        mean_error = historic_errors.mean()
        # Fallback to standard deviation if mean error is 0
        error_buffer = mean_error if mean_error > 0 else historic_errors.std()
    else:
        # Safety fallback default if columns are misspelled
        error_buffer = 150.0

    df['Units_Sold'] = df['Units_Sold'].fillna(0)
    if 'Units_Ordered' in df.columns:
        df['Units_Ordered'] = df['Units_Ordered'].fillna(0)
    if 'Demand' in df.columns:
        df['Demand'] = df['Demand'].fillna(0)


        # 5. GENERATE YOUR MODEL FORECAST
    # (Assuming you pass this into your XGBoost/Prophet pipeline.
    # For this illustration, we generate a mock projection based on your dynamic history)
    history_dates = df['Date'].astype(str).tolist()
    history_actuals = df['Units_Sold'].tolist() if 'Units_Sold' in df.columns else df['Demand'].tolist()

    # Mocking a 30-day projection that scales with your seasonality choices
    forecast_dates = []
    forecast_predictions = []
    lower_bounds = []
    upper_bounds = []

    last_date = pd.to_datetime(history_dates[-1])
    last_val = history_actuals[-1]

    for i in range(1, 31):
        next_date = (last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        forecast_dates.append(next_date)

        # Apply seasonality adjustments based on user input string
        season_effect = np.sin(i / 3.5) * (50 if seasonality_mode == "additive" else (last_val * 0.05))
        pred = max(10, int(last_val + (i * 0.5) + season_effect))
        forecast_predictions.append(pred)

        # Widen the band dynamically over time using your computed historic error buffer
        # and the user's custom confidence multiplier parameter
        dynamic_spread = (error_buffer * confidence_multiplier) + (i * 2)
        lower_bounds.append(max(0, int(pred - dynamic_spread)))
        upper_bounds.append(int(pred + dynamic_spread))

    # 6. RETURN PAYLOAD structured perfectly for your Plotly dashboard
    return {
        "dates": history_dates + forecast_dates,
        "actual": history_actuals + [None] * 30,
        "predicted": [None] * len(history_dates) + forecast_predictions,
        "lower_bound": [None] * len(history_dates) + lower_bounds,
        "upper_bound": [None] * len(history_dates) + upper_bounds
    }


@app.get("/api/anomalies")
def get_anomalies(
        region: str = Query("All"),
        days_history: int = Query(90),
        # Drop threshold down from 2.0 to 0.5 to intentionally catch anomalies for display!
        anomaly_threshold: float = Query(0.5)
):
    df = load_and_clean_data('demand_forecasting.csv')
    df_feat = build_xgboost_features(df, target_col='Units_Sold')
    
    if region != "All" and "Region" in df.columns:
        df = df[df['Region'] == region]

    df = df.tail(days_history).reset_index(drop=True)

    anomalies_list = []

    if 'Units_Ordered' in df.columns and 'Demand' in df.columns:
        # Calculate standard historic differences
        df['difference'] = (df['Units_Ordered'] - df['Demand']).abs()
        avg_diff = df['difference'].mean()
        std_diff = df['difference'].std() if df['difference'].std() > 0 else 1.0

        # Flag rows where execution diverged from requirements
        for idx, row in df.iterrows():
            if row['difference'] > (avg_diff + (anomaly_threshold * std_diff)):
                severity = "CRITICAL" if row['difference'] > (avg_diff * 1.5) else "WARNING"

                anomalies_list.append({
                    "date": str(row['Date'])[:10],
                    "value": int(row['Demand']),
                    "expected": int(row['Units_Ordered']),
                    "severity": severity,
                    "description": f"Supply gap mismatch of {int(row['difference'])} units."
                })

    # ─── CRUCIAL FALLBACK SAFETY NET ─────────────────────────────────────
    # If the list is completely empty, inject a friendly informational record
    # instead of sending nothing, preventing the frontend from crashing!
    if not anomalies_list:
        anomalies_list.append({
            "date": "N/A",
            "value": 0,
            "expected": 0,
            "severity": "INFO",
            "description": "System operational. No critical inventory anomalies detected in this subset."
        })

    return {"anomalies_detected": anomalies_list}