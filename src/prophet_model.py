from prophet import Prophet
import pandas as pd

class ProphetForecaster:
    def __init__(self):
        # Configure for business data (include weekly/yearly patterns)
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        
    def fit(self, df, date_col, target_col):
        # Rename columns to match Prophet's requirements
        prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        self.model.fit(prophet_df)
        
    def predict(self, periods=30):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        # Returns expected value (yhat) along with uncertainty bounds
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
