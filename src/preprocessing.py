import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def build_xgboost_features(df, target_col):
    """Creates lag and rolling features for tabular models."""
    df_feat = df.copy()

    # Time-based categories
    df_feat['day_of_week'] = df_feat['Date'].dt.dayofweek
    df_feat['month_num'] = df_feat['Date'].dt.month
    
    # Convert to string so that it can be strftime can be used
    df_feat['Date'] = df['Date'].astype(str)
    # Lag Features (Looking backward)
    df_feat['lag_1'] = df_feat[target_col].shift(1)
    df_feat['lag_7'] = df_feat[target_col].shift(7)
    df_feat['lag_30'] = df_feat[target_col].shift(30)
    
    # Rolling Statistics
    df_feat['rolling_mean_7'] = df_feat[target_col].shift(1).rolling(window=7).mean()
    
    return df_feat.dropna()

def prepare_lstm_sequences(data, sequence_length):
    """Converts 2D array into a 3D tensor format required by LSTMs."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

