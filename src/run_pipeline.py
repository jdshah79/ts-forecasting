from preprocessing import load_and_clean_data, build_xgboost_features, prepare_lstm_sequences
import pandas as pd
import numpy as np

# ── Step 1: Load and clean ──────────────────────────────
df = load_and_clean_data('demand_forecasting.csv')
print(df.head())
print(df.dtypes)

# ── Step 2: Build XGBoost features ─────────────────────
# target_col is 'units_sold=sales' — this adds lag and rolling features
df_features = build_xgboost_features(df, target_col='Units_Sold')
print(df_features.head())

# Split into X (features) and y (target)
feature_cols = ['day_of_week', 'Store_ID', 'Product_ID', 'Category', 'Region', 'Inventory_Level', 'Units_Sold', 'Units_Ordered', 'Price', 'Discount', 'Weather_Condition', 'Promotion', 'Competitor_Pricing', 'Seasonality', 'Epidemic', 'Demand']  # add any extra columns you want

X = df_features[feature_cols].values
y = df_features['Units_Sold'].values

# Train/val split (80/20)
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ── Step 3: Prepare LSTM sequences ─────────────────────
# Normalise first — LSTM expects values roughly in 0-1 range
sales = df['Units_Sold'].values.reshape(-1, 1)
sales_min, sales_max = sales.min(), sales.max()
sales_norm = (sales - sales_min) / (sales_max - sales_min)

SEQUENCE_LENGTH = 30  # use 30 days of history to predict the next day
X_lstm, y_lstm = prepare_lstm_sequences(sales_norm, SEQUENCE_LENGTH)

# Reshape for PyTorch: (samples, sequence_length, features)
X_lstm = X_lstm.reshape(X_lstm.shape[0], SEQUENCE_LENGTH, 1)

print(f"XGBoost — X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"LSTM    — X: {X_lstm.shape}, y: {y_lstm.shape}")
