def blend_predictions(prophet_preds, xgb_preds, lstm_preds):
    """
    Blends predictions. Weights can be adjusted based on validation error (RMSE).
    For example: XGBoost handles spikes well, Prophet handles baselines well.
    """
    final_forecast = (0.2 * prophet_preds) + (0.5 * xgb_preds) + (0.3 * lstm_preds)
    return final_forecast

def detect_anomalies(actual_df, prophet_forecast_df):
    """
    Merges actual metrics with Prophet's upper/lower confidence bands.
    Flags instances where the actual value drops below or spikes above the bands.
    """
    # Merge on date
    merged = pd.merge(actual_df, prophet_forecast_df, on='ds')
    
    # Anomaly flag logic
    merged['is_anomaly'] = (merged['actual'] > merged['yhat_upper']) | \
                           (merged['actual'] < merged['yhat_lower'])
                           
    return merged[merged['is_anomaly'] == True]


