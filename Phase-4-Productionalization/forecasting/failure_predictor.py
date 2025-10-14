import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

class FailurePredictor:
    """
    Predict system failures based on metric patterns
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.prediction_horizon = 2  # Predict failures 2 hours ahead
        
    def create_failure_labels(self, historical_data, failure_thresholds=None):
        """Create failure labels based on extreme metric values"""
        if failure_thresholds is None:
            failure_thresholds = {
                'cpu_percent': 95,
                'memory_percent': 98, 
                'disk_usage_percent': 99
            }
        
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Label as failure if any metric exceeds threshold in next 2 hours (24 periods of 5min)
        df['future_max_cpu'] = df['cpu_percent'].rolling(window=24, min_periods=1).max().shift(-24)
        df['future_max_memory'] = df['memory_percent'].rolling(window=24, min_periods=1).max().shift(-24)
        df['future_max_disk'] = df['disk_usage_percent'].rolling(window=24, min_periods=1).max().shift(-24)
        
        # Failure if any future metric exceeds threshold
        df['will_fail'] = (
            (df['future_max_cpu'] > failure_thresholds['cpu_percent']) |
            (df['future_max_memory'] > failure_thresholds['memory_percent']) | 
            (df['future_max_disk'] > failure_thresholds['disk_usage_percent'])
        ).astype(int)
        
        return df.dropna()
    
    def prepare_failure_features(self, df):
        """Prepare features for failure prediction"""
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Recent statistics (last hour)
        df['cpu_rolling_mean_12'] = df['cpu_percent'].rolling(window=12, min_periods=1).mean()
        df['cpu_rolling_std_12'] = df['cpu_percent'].rolling(window=12, min_periods=1).std()
        df['cpu_rolling_max_12'] = df['cpu_percent'].rolling(window=12, min_periods=1).max()
        
        df['memory_rolling_mean_12'] = df['memory_percent'].rolling(window=12, min_periods=1).mean()
        df['memory_rolling_std_12'] = df['memory_percent'].rolling(window=12, min_periods=1).std()
        df['memory_rolling_max_12'] = df['memory_percent'].rolling(window=12, min_periods=1).max()
        
        # Rate of change
        df['cpu_roc_1'] = df['cpu_percent'].pct_change().fillna(0)
        df['memory_roc_1'] = df['memory_percent'].pct_change().fillna(0)
        
        # Volatility
        df['cpu_volatility'] = df['cpu_rolling_std_12'] / (df['cpu_rolling_mean_12'] + 1e-8)
        df['memory_volatility'] = df['memory_rolling_std_12'] / (df['memory_rolling_mean_12'] + 1e-8)
        
        # Resource pressure indicator
        df['resource_pressure'] = (
            (df['cpu_percent'] > 80).astype(int) + 
            (df['memory_percent'] > 85).astype(int) +
            (df['disk_usage_percent'] > 90).astype(int)
        )
        
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend',
            'cpu_percent', 'memory_percent', 'disk_usage_percent',
            'cpu_rolling_mean_12', 'cpu_rolling_std_12', 'cpu_rolling_max_12',
            'memory_rolling_mean_12', 'memory_rolling_std_12', 'memory_rolling_max_12', 
            'cpu_roc_1', 'memory_roc_1',
            'cpu_volatility', 'memory_volatility',
            'resource_pressure'
        ]
        
        return df, feature_columns
    
    def predict_failure_risk(self, recent_metrics):
        """Predict failure risk in the next 2 hours"""
        if self.model is None:
            return {"error": "Model not trained yet"}
        
        # Prepare current state
        current_data = pd.DataFrame(recent_metrics[-12:])  # Last hour
        current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
        current_data = current_data.sort_values('timestamp')
        
        if len(current_data) == 0:
            return {"error": "Insufficient data"}
        
        # Get latest point and calculate features
        latest = current_data.iloc[-1].copy()
        
        features = {}
        
        # Current metrics
        features['cpu_percent'] = latest['cpu_percent']
        features['memory_percent'] = latest['memory_percent']
        features['disk_usage_percent'] = latest.get('disk_usage_percent', 50)  # Default if missing
        
        # Time features
        timestamp = datetime.now()
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        
        # Rolling statistics (last hour)
        features['cpu_rolling_mean_12'] = current_data['cpu_percent'].mean()
        features['cpu_rolling_std_12'] = current_data['cpu_percent'].std() if len(current_data) > 1 else 0
        features['cpu_rolling_max_12'] = current_data['cpu_percent'].max()
        
        features['memory_rolling_mean_12'] = current_data['memory_percent'].mean()
        features['memory_rolling_std_12'] = current_data['memory_percent'].std() if len(current_data) > 1 else 0
        features['memory_rolling_max_12'] = current_data['memory_percent'].max()
        
        # Rate of change
        if len(current_data) >= 2:
            features['cpu_roc_1'] = (current_data['cpu_percent'].iloc[-1] - current_data['cpu_percent'].iloc[-2]) / current_data['cpu_percent'].iloc[-2]
            features['memory_roc_1'] = (current_data['memory_percent'].iloc[-1] - current_data['memory_percent'].iloc[-2]) / current_data['memory_percent'].iloc[-2]
        else:
            features['cpu_roc_1'] = 0
            features['memory_roc_1'] = 0
        
        # Volatility
        features['cpu_volatility'] = features['cpu_rolling_std_12'] / (features['cpu_rolling_mean_12'] + 1e-8)
        features['memory_volatility'] = features['memory_rolling_std_12'] / (features['memory_rolling_mean_12'] + 1e-8)
        
        # Resource pressure
        features['resource_pressure'] = (
            (features['cpu_percent'] > 80) + 
            (features['memory_percent'] > 85) +
            (features['disk_usage_percent'] > 90)
        )
        
        # Create feature vector
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        
        # Make prediction
        try:
            # FIX: Handle different model types
            if hasattr(self.model, 'predict_proba'):
                failure_prob = self.model.predict_proba([feature_vector])[0][1]
            else:
                # For fallback DummyClassifier, use a low probability
                failure_prob = 0.1  # Low probability for stable systems
            
            will_fail = failure_prob > 0.5
            
            return {
                'failure_probability': failure_prob,
                'will_fail': bool(will_fail),
                'confidence': failure_prob if will_fail else 1 - failure_prob,
                'time_horizon': '2 hours',
                'risk_level': self._get_risk_level(failure_prob)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium" 
        else:
            return "high"
    
    def generate_failure_alerts(self, prediction):

        """Generate alerts based on failure prediction"""
        if "error" in prediction:
            return []
        
        alerts = []
        prob = prediction['failure_probability']
        risk_level = prediction['risk_level']
        
        if risk_level == "high" and prob > 0.8:
            alerts.append({
                "level": "critical",
                "title": "🚨 IMMINENT SYSTEM FAILURE PREDICTED",
                "message": f"High probability ({prob:.1%}) of system failure within {prediction['time_horizon']}. Take immediate action.",
                "probability": prob,
                "time_horizon": prediction['time_horizon'],
                "confidence": prediction['confidence']
            })
        elif risk_level == "medium" and prob > 0.5:
            alerts.append({
                "level": "warning",
                "title": "⚠️ Potential System Issue Detected",
                "message": f"Moderate probability ({prob:.1%}) of system issues within {prediction['time_horizon']}. Monitor closely.",
                "probability": prob, 
                "time_horizon": prediction['time_horizon'],
                "confidence": prediction['confidence']
            })
        
        return alerts

    def train_failure_model(self, historical_data):
        """Train failure prediction model"""
        print("🔮 Training Failure Prediction Model...")
    
        # Create labeled data
        labeled_data = self.create_failure_labels(historical_data)
    
        # Prepare features
        prepared_data, feature_columns = self.prepare_failure_features(labeled_data)
        self.feature_columns = feature_columns

        # Split features and target
        X = prepared_data[feature_columns].fillna(0)
        y = prepared_data['will_fail']
        
        print(f"   Dataset: {len(X)} samples, {y.sum()} failure events ({y.mean():.2%})")
        
        # FIX: Handle case with no failure events
        if y.sum() == 0:
            print("   ⚠️  No failure events in training data. Using fallback model.")
            # Create a simple model that always predicts no failure
            from sklearn.dummy import DummyClassifier
            self.model = DummyClassifier(strategy='constant', constant=0)
            self.model.fit(X, y)
            print("   ✅ Fallback model trained (always predicts stable)")
            return True
        
        # Handle class imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y), y=y
        )
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("   📊 Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Stable', 'Failure']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("   Confusion Matrix:")
        print(f"   {cm}")
        
        # Calculate precision for failure class
        failure_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        print(f"   🎯 Failure Precision: {failure_precision:.3f}")
        
        return True
#