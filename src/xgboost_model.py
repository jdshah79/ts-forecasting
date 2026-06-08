import xgboost as xgb

class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
    def predict(self, X):
        return self.model.predict(X)
