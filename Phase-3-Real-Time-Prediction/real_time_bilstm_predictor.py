import json
import time
import redis
import psycopg2
import joblib
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append('..')

class BiLSTMPredictor:
    def __init__(self):
        # Load the trained BiLSTM model and components
        try:
            # Load PyTorch model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🚀 Using device: {self.device}")
            
            # Load feature engineering components
            self.scaler = joblib.load('../Phase-2-Anomaly-Detection/scaler_pytorch.pkl')
            self.feature_names = joblib.load('../Phase-2-Anomaly-Detection/feature_names_pytorch.pkl')
            
            # Recreate the BiLSTM model architecture
            self.sequence_length = 10
            self.model = self.create_bilstm_model(len(self.feature_names))
            
            # Load trained weights
            model_path = '../Phase-2-Anomaly-Detection/best_bilstm_model.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()  # Set to evaluation mode
                print("✅ Loaded trained BiLSTM model with temporal pattern recognition")
            else:
                raise FileNotFoundError("BiLSTM model weights not found")
                
        except Exception as e:
            print(f"❌ Error loading BiLSTM model: {e}")
            sys.exit(1)
        
        # Setup Redis
        self.redis_client = self.setup_redis()
        
        # Store recent metrics for sequence creation
        self.sequence_buffer = []
        self.max_sequence_length = 20  # Keep more than needed for flexibility
        
        # Alert management
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.consecutive_anomalies = 0
        self.anomaly_threshold = 3  # Number of consecutive anomalies to trigger alert
        
        print("🧠 BiLSTM Real-Time Predictor Initialized!")
        print("   - Temporal pattern recognition enabled")
        print("   - Sequence length: 10 time steps")
        print("   - Early failure detection capability")
    
    def create_bilstm_model(self, input_size):
        """Recreate the BiLSTM model architecture"""
        import torch.nn as nn
        
        class BiLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
                super(BiLSTMModel, self).__init__()
                
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True, 
                    bidirectional=True,
                    dropout=dropout
                )
                
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                lstm_out, (hidden, cell) = self.lstm(x)
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                output = self.classifier(context_vector)
                return output
        
        return BiLSTMModel(input_size).to(self.device)
    
    def setup_redis(self):
        """Setup Redis connection"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
            r.ping()
            print("✅ Connected to Redis successfully")
            return r
        except redis.ConnectionError:
            print("❌ Could not connect to Redis")
            sys.exit(1)
    
    def enhanced_feature_engineering_realtime(self, current_metric):
        """Create features matching the training pipeline"""
        features = {}
        
        # Basic metrics
        base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
        for feature in base_features:
            features[feature] = current_metric[feature]
        
        # Time-based features
        current_time = datetime.now()
        features['hour'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['is_weekend'] = 1 if current_time.weekday() >= 5 else 0
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Add current metric to sequence buffer
        self.sequence_buffer.append(current_metric)
        if len(self.sequence_buffer) > self.max_sequence_length:
            self.sequence_buffer.pop(0)
        
        # Calculate rolling features if we have enough data
        if len(self.sequence_buffer) >= 5:
            df = pd.DataFrame(self.sequence_buffer)
            
            for feature in base_features:
                # Rolling statistics
                features[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean().iloc[-1]
                features[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std().iloc[-1]
                features[f'{feature}_rolling_min_5'] = df[feature].rolling(window=5, min_periods=1).min().iloc[-1]
                features[f'{feature}_rolling_max_5'] = df[feature].rolling(window=5, min_periods=1).max().iloc[-1]
                
                # Rate of change
                if len(df) >= 2:
                    features[f'{feature}_roc'] = df[feature].pct_change().fillna(0).iloc[-1]
        
        # Interaction features
        features['cpu_memory_ratio'] = current_metric['cpu_percent'] / (current_metric['memory_percent'] + 1)
        features['system_load_index'] = (current_metric['cpu_percent'] * 0.6 + 
                                       current_metric['memory_percent'] * 0.3 + 
                                       current_metric['disk_usage_percent'] * 0.1)
        features['resource_pressure'] = (
            (current_metric['cpu_percent'] > 80).astype(int) + 
            (current_metric['memory_percent'] > 80).astype(int) + 
            (current_metric['disk_usage_percent'] > 80).astype(int)
        )
        
        # Fill missing features with 0
        for feature_name in self.feature_names:
            if feature_name not in features:
                features[feature_name] = 0
        
        # Create feature vector in exact same order as training
        feature_vector = [features[feature_name] for feature_name in self.feature_names]
        
        return np.array(feature_vector)
    
    def create_sequence_for_prediction(self):
        """Create sequence for BiLSTM prediction"""
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # Get the most recent sequence
        recent_sequence = self.sequence_buffer[-self.sequence_length:]
        
        # Create feature vectors for each point in sequence
        sequence_features = []
        for metric in recent_sequence:
            features = self.enhanced_feature_engineering_realtime(metric)
            sequence_features.append(features)
        
        return np.array(sequence_features)
    
    def predict_anomaly(self, sequence):
        """Predict anomaly using BiLSTM model"""
        try:
            # Scale the sequence
            sequence_scaled = self.scaler.transform(sequence.reshape(-1, len(self.feature_names)))
            sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, -1)
            
            # Convert to PyTorch tensor
            sequence_tensor = torch.FloatTensor(sequence_scaled).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probability = output.cpu().numpy()[0][0]
                is_anomaly = probability > 0.5
            
            return is_anomaly, probability, sequence_scaled
            
        except Exception as e:
            print(f"❌ BiLSTM prediction error: {e}")
            return False, 0.0, None
    
    def analyze_temporal_pattern(self, sequence_scaled):
        """Analyze temporal patterns for better alerting"""
        if sequence_scaled is None:
            return "Unknown pattern"
        
        # Analyze trends in the sequence
        sequence_df = pd.DataFrame(sequence_scaled[0], columns=self.feature_names)
        
        patterns = []
        
        # Check CPU trend
        cpu_trend = np.polyfit(range(len(sequence_df)), sequence_df['cpu_percent'], 1)[0]
        if cpu_trend > 0.5:
            patterns.append("Rising CPU trend")
        elif cpu_trend < -0.5:
            patterns.append("Falling CPU trend")
        
        # Check memory trend
        memory_trend = np.polyfit(range(len(sequence_df)), sequence_df['memory_percent'], 1)[0]
        if memory_trend > 0.3:
            patterns.append("Memory leak pattern")
        
        # Check for spikes
        cpu_std = sequence_df['cpu_percent'].std()
        if cpu_std > 15:
            patterns.append("CPU volatility")
        
        if not patterns:
            patterns.append("Complex temporal pattern")
        
        return " | ".join(patterns)
    
    def should_alert(self, alert_type, current_time, probability):
        """Enhanced alert logic with temporal awareness"""
        # Reset consecutive count if no anomaly
        if probability < 0.3:
            self.consecutive_anomalies = 0
            return False
        
        # Increment consecutive anomalies
        self.consecutive_anomalies += 1
        
        # Check cooldown
        if alert_type in self.last_alert_time:
            time_since_last = (current_time - self.last_alert_time[alert_type]).total_seconds()
            if time_since_last < self.alert_cooldown:
                return False
        
        # Only alert after consecutive anomalies or high probability
        return self.consecutive_anomalies >= self.anomaly_threshold or probability > 0.8
    
    def create_intelligent_alert(self, data, probability, temporal_pattern):
        """Create intelligent alert with temporal insights"""
        current_time = datetime.now()
        
        alert_level = "🚨 CRITICAL" if probability > 0.8 else "⚠️  WARNING"
        
        return f"""
{alert_level} - SYSTEM ANOMALY DETECTED

📊 Temporal Analysis:
{temporal_pattern}

🎯 Confidence: {probability:.1%}
📈 Consecutive anomalies: {self.consecutive_anomalies}
🕒 Detection time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}

📋 Current State:
- CPU Usage: {data['cpu_percent']}%
- Memory Usage: {data['memory_percent']}% 
- Disk Usage: {data['disk_usage_percent']}%
- Memory Used: {data['memory_used_gb']} GB

💡 Recommended Actions:
- Review system logs for errors
- Check for resource leaks
- Monitor application performance
- Consider scaling resources

🔍 Detection Method: BiLSTM Temporal Pattern Recognition
"""

    def start_bilstm_monitoring(self):
        """Start real-time monitoring with BiLSTM"""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe('metrics')
        
        print("\n🧠 BI-LSTM TEMPORAL MONITORING STARTED!")
        print("⭐ Using PyTorch BiLSTM with Attention Mechanism")
        print("📊 Monitoring temporal patterns across 10 time steps")
        print("🔔 Smart alerting with consecutive anomaly detection")
        print("Press Ctrl+C to stop\n")
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        
                        # Create sequence for prediction
                        sequence = self.create_sequence_for_prediction()
                        
                        if sequence is not None:
                            # Make prediction with BiLSTM
                            is_anomaly, probability, sequence_scaled = self.predict_anomaly(sequence)
                            
                            # Analyze temporal pattern
                            temporal_pattern = self.analyze_temporal_pattern(sequence_scaled)
                            
                            # Display result with temporal insights
                            status = "🧠 ANOMALY" if is_anomaly else "✅ Normal"
                            trend_icon = "📈" if "Rising" in temporal_pattern else "📉" if "Falling" in temporal_pattern else "📊"
                            
                            print(f"{status} | Conf: {probability:.3f} | Seq: {len(self.sequence_buffer)} | "
                                  f"{trend_icon} {temporal_pattern[:30]}... | "
                                  f"CPU: {data['cpu_percent']}% | Time: {data['timestamp'][11:19]}")
                            
                            # Trigger intelligent alert
                            current_time = datetime.now()
                            if is_anomaly and self.should_alert("system_anomaly", current_time, probability):
                                alert_message = self.create_intelligent_alert(data, probability, temporal_pattern)
                                self.send_alert(alert_message)
                                self.last_alert_time["system_anomaly"] = current_time
                                
                        else:
                            # Still building sequence
                            print(f"🔄 Building sequence: {len(self.sequence_buffer)}/{self.sequence_length} | "
                                  f"CPU: {data['cpu_percent']}% | Time: {data['timestamp'][11:19]}")
                            
                    except Exception as e:
                        print(f"⚠️  Processing error: {e}")
                        
        except KeyboardInterrupt:
            print("\n🛑 BiLSTM monitoring stopped by user")
    
    def send_alert(self, message):
        """Send alert to console"""
        print("\n" + "="*70)
        print("🧠 BI-LSTM INTELLIGENT ALERT")
        print("="*70)
        print(message)
        print("="*70 + "\n")

if __name__ == '__main__':
    predictor = BiLSTMPredictor()
    predictor.start_bilstm_monitoring()