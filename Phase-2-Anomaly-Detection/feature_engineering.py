import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_features(df):
    """Create time-series features for anomaly detection"""
    features_df = df.copy()
    
    # Basic features
    features = [
        'cpu_percent', 
        'memory_percent', 
        'memory_used_gb',
        'disk_usage_percent',
        'disk_used_gb'
    ]
    
    # Create rolling statistics (time-based features)
    for feature in features:
        # Rolling mean (short-term average)
        features_df[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
        
        # Rolling standard deviation (volatility)
        features_df[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std()
        
        # Rate of change
        features_df[f'{feature}_roc'] = df[feature].pct_change().fillna(0)
        
        # Z-score (how many standard deviations from mean)
        features_df[f'{feature}_zscore'] = (df[feature] - df[feature].mean()) / df[feature].std()
    
    # Time-based features
    features_df['hour'] = df['timestamp'].dt.hour
    features_df['day_of_week'] = df['timestamp'].dt.dayofweek
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    
    # Fill any NaN values created by rolling windows
    features_df = features_df.fillna(method='bfill').fillna(method='ffill')
    
    print(f"✅ Created {len([col for col in features_df.columns if col not in ['timestamp', 'is_anomaly']])} features")
    
    return features_df

def prepare_training_data(features_df):
    """Prepare features and labels for training"""
    # Select feature columns (exclude timestamp and target)
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns]
    y = features_df['is_anomaly']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"🎯 Anomaly rate in training set: {y.mean():.2%}")
    
    return X_scaled, y, scaler, feature_columns

if __name__ == '__main__':
    # Test the feature engineering
    df = pd.read_csv('training_dataset.csv', parse_dates=['timestamp'])
    features_df = create_features(df)
    X, y, scaler, feature_columns = prepare_training_data(features_df)
    
    print(f"\n📋 Feature columns: {feature_columns}")