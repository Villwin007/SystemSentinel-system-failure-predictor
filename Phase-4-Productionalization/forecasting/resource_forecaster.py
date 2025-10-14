import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ResourceForecaster:
    """
    Predict future system resource usage (CPU, Memory, Disk)
    """
    
    def __init__(self):
        self.models = {}
        self.forecast_horizon = 6  # Predict 6 hours ahead
        self.feature_columns = []
        
    def prepare_forecasting_data(self, historical_data):
        """Prepare data for forecasting model"""
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12]:  # 5min intervals, so 1=5min, 12=1hour
            df[f'cpu_lag_{lag}'] = df['cpu_percent'].shift(lag)
            df[f'memory_lag_{lag}'] = df['memory_percent'].shift(lag)
        
        # Rolling statistics
        df['cpu_rolling_mean_6'] = df['cpu_percent'].rolling(window=6, min_periods=1).mean()
        df['cpu_rolling_std_6'] = df['cpu_percent'].rolling(window=6, min_periods=1).std()
        df['memory_rolling_mean_6'] = df['memory_percent'].rolling(window=6, min_periods=1).mean()
        
        # Target: future values (6 hours ahead = 72 periods of 5min each)
        df['cpu_future_72'] = df['cpu_percent'].shift(-72)
        df['memory_future_72'] = df['memory_percent'].shift(-72)
        df['disk_future_72'] = df['disk_usage_percent'].shift(-72)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_forecasting_models(self, historical_data):
        """Train forecasting models for CPU, Memory, and Disk"""
        print("🧠 Training Resource Forecasting Models...")
        
        prepared_data = self.prepare_forecasting_data(historical_data)
        
        # Define features
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos',
            'cpu_lag_1', 'cpu_lag_2', 'cpu_lag_3', 'cpu_lag_6', 'cpu_lag_12',
            'memory_lag_1', 'memory_lag_2', 'memory_lag_3', 'memory_lag_6', 'memory_lag_12',
            'cpu_rolling_mean_6', 'cpu_rolling_std_6', 'memory_rolling_mean_6'
        ]
        
        self.feature_columns = feature_columns
        
        targets = {
            'cpu': 'cpu_future_72',
            'memory': 'memory_future_72', 
            'disk': 'disk_future_72'
        }
        
        for resource, target_col in targets.items():
            print(f"   Training {resource.upper()} forecaster...")
            
            X = prepared_data[feature_columns]
            y = prepared_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"     ✅ {resource.upper()} - MAE: {mae:.2f}%, RMSE: {rmse:.2f}%")
            
            self.models[resource] = model
        
        print("🎯 Forecasting models trained successfully!")
        return True
    
    def predict_future_resources(self, recent_metrics):
        """Predict resource usage 6 hours into the future"""
        if not self.models:
            return {"error": "Models not trained yet"}
        
        # Prepare current state for prediction
        current_data = pd.DataFrame(recent_metrics[-72:])  # Last 6 hours
        current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
        current_data = current_data.sort_values('timestamp')
        
        # Get the most recent point
        latest = current_data.iloc[-1].copy()
        
        # Create features for prediction
        features = {}
        timestamp = datetime.now()
        
        # Time features
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Lag features (using recent history)
        if len(current_data) >= 12:
            features['cpu_lag_1'] = current_data['cpu_percent'].iloc[-1]
            features['cpu_lag_2'] = current_data['cpu_percent'].iloc[-2] if len(current_data) >= 2 else latest['cpu_percent']
            features['cpu_lag_3'] = current_data['cpu_percent'].iloc[-3] if len(current_data) >= 3 else latest['cpu_percent']
            features['cpu_lag_6'] = current_data['cpu_percent'].iloc[-6] if len(current_data) >= 6 else latest['cpu_percent']
            features['cpu_lag_12'] = current_data['cpu_percent'].iloc[-12] if len(current_data) >= 12 else latest['cpu_percent']
            
            features['memory_lag_1'] = current_data['memory_percent'].iloc[-1]
            features['memory_lag_2'] = current_data['memory_percent'].iloc[-2] if len(current_data) >= 2 else latest['memory_percent']
            features['memory_lag_3'] = current_data['memory_percent'].iloc[-3] if len(current_data) >= 3 else latest['memory_percent']
            features['memory_lag_6'] = current_data['memory_percent'].iloc[-6] if len(current_data) >= 6 else latest['memory_percent']
            features['memory_lag_12'] = current_data['memory_percent'].iloc[-12] if len(current_data) >= 12 else latest['memory_percent']
        else:
            # If not enough history, use current values
            for lag in [1, 2, 3, 6, 12]:
                features[f'cpu_lag_{lag}'] = latest['cpu_percent']
                features[f'memory_lag_{lag}'] = latest['memory_percent']
        
        # Rolling statistics
        features['cpu_rolling_mean_6'] = current_data['cpu_percent'].mean() if len(current_data) > 0 else latest['cpu_percent']
        features['cpu_rolling_std_6'] = current_data['cpu_percent'].std() if len(current_data) > 1 else 0
        features['memory_rolling_mean_6'] = current_data['memory_percent'].mean() if len(current_data) > 0 else latest['memory_percent']
        
        # Create feature vector
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        
        # Make predictions
        predictions = {}
        for resource, model in self.models.items():
            try:
                pred = model.predict([feature_vector])[0]
                predictions[f'{resource}_6h'] = max(0, min(100, pred))  # Clamp to 0-100%
            except Exception as e:
                predictions[f'{resource}_6h'] = None
        
        # Add confidence scores based on model performance
        predictions['confidence'] = 0.85  # Placeholder confidence
        
        return predictions
    
    def generate_forecast_alerts(self, predictions, current_metrics):
        """Generate alerts based on forecasted values"""
        alerts = []
        
        cpu_6h = predictions.get('cpu_6h')
        memory_6h = predictions.get('memory_6h')
        disk_6h = predictions.get('disk_6h')
        
        if cpu_6h and cpu_6h > 90:
            alerts.append({
                "level": "warning",
                "title": "CPU Usage Forecast Alert",
                "message": f"CPU usage predicted to reach {cpu_6h:.1f}% in 6 hours. Consider optimizing workloads.",
                "predicted_value": cpu_6h,
                "time_horizon": "6 hours",
                "confidence": predictions.get('confidence', 0.0)
            })
        
        if memory_6h and memory_6h > 85:
            alerts.append({
                "level": "warning", 
                "title": "Memory Usage Forecast Alert",
                "message": f"Memory usage predicted to reach {memory_6h:.1f}% in 6 hours. Monitor for memory leaks.",
                "predicted_value": memory_6h,
                "time_horizon": "6 hours",
                "confidence": predictions.get('confidence', 0.0)
            })
        
        if disk_6h and disk_6h > 90:
            alerts.append({
                "level": "critical",
                "title": "Disk Space Forecast Alert", 
                "message": f"Disk usage predicted to reach {disk_6h:.1f}% in 6 hours. Free up disk space immediately.",
                "predicted_value": disk_6h,
                "time_horizon": "6 hours", 
                "confidence": predictions.get('confidence', 0.0)
            })
        
        return alerts