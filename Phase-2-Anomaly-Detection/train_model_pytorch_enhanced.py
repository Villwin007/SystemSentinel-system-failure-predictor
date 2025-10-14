# 
# TENSORFLOW ON CPU
# 
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier, IsolationForest
# from sklearn.svm import OneClassSVM
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
# warnings.filterwarnings('ignore')

# # Set random seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# def create_sequences(data, labels, sequence_length=10):
#     """Create sequences for LSTM training"""
#     X_sequences = []
#     y_sequences = []
    
#     for i in range(len(data) - sequence_length):
#         X_sequences.append(data[i:i + sequence_length])
#         y_sequences.append(labels[i + sequence_length - 1])  # Predict last point in sequence
    
#     return np.array(X_sequences), np.array(y_sequences)

# def build_bilstm_model(input_shape, num_features):
#     """Build BiLSTM model for anomaly detection"""
#     model = Sequential([
#         # Conv1D for local pattern detection
#         Conv1D(filters=64, kernel_size=3, activation='relu', 
#                input_shape=input_shape, padding='same'),
#         MaxPooling1D(pool_size=2),
        
#         # Bidirectional LSTM for temporal patterns
#         Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
#         Bidirectional(LSTM(32, dropout=0.2)),
        
#         # Dense layers for classification
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')  # Binary classification
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=['accuracy', 'precision', 'recall']
#     )
    
#     return model

# def train_bilstm(X_train, y_train, X_test, y_test, feature_names, sequence_length=10):
#     """Train BiLSTM model on sequential data"""
#     print("🧠 Training BiLSTM Model...")
    
#     # Create sequences
#     X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
#     X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
#     print(f"   Training sequences: {X_train_seq.shape}")
#     print(f"   Test sequences: {X_test_seq.shape}")
    
#     # Build model
#     input_shape = (sequence_length, X_train.shape[1])
#     model = build_bilstm_model(input_shape, X_train.shape[1])
    
#     print("   Model architecture:")
#     model.summary()
    
#     # Callbacks
#     callbacks = [
#         EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
#     ]
    
#     # Train model
#     history = model.fit(
#         X_train_seq, y_train_seq,
#         batch_size=32,
#         epochs=50,
#         validation_data=(X_test_seq, y_test_seq),
#         callbacks=callbacks,
#         verbose=1,
#         shuffle=True
#     )
    
#     # Make predictions
#     y_pred_proba = model.predict(X_test_seq)
#     y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
#     # Calculate metrics
#     accuracy = np.mean(y_pred == y_test_seq)
    
#     print(f"✅ BiLSTM trained with {accuracy:.2%} accuracy on test sequences")
    
#     # Plot training history
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('bilstm_training_history.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return model, y_pred, y_test_seq, history

# def enhanced_feature_engineering(df):
#     """Create robust features for all models"""
#     features_df = df.copy()
    
#     base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
    
#     # Time-based features
#     features_df['hour'] = df['timestamp'].dt.hour
#     features_df['day_of_week'] = df['timestamp'].dt.dayofweek
#     features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
#     features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
#     features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
#     # Rolling statistics
#     for feature in base_features:
#         # Basic rolling features
#         features_df[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
#         features_df[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std()
#         features_df[f'{feature}_rolling_min_5'] = df[feature].rolling(window=5, min_periods=1).min()
#         features_df[f'{feature}_rolling_max_5'] = df[feature].rolling(window=5, min_periods=1).max()
        
#         # Rate of change
#         features_df[f'{feature}_roc'] = df[feature].pct_change().fillna(0)
        
#         # Volatility
#         rolling_mean = df[feature].rolling(window=10, min_periods=1).mean()
#         rolling_std = df[feature].rolling(window=10, min_periods=1).std()
#         features_df[f'{feature}_volatility'] = rolling_std / (rolling_mean + 1e-8)
    
#     # Interaction features
#     features_df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_percent'] + 1)
#     features_df['system_load_index'] = (df['cpu_percent'] * 0.6 + df['memory_percent'] * 0.3 + df['disk_usage_percent'] * 0.1)
#     features_df['resource_pressure'] = (df['cpu_percent'] > 80).astype(int) + (df['memory_percent'] > 80).astype(int) + (df['disk_usage_percent'] > 80).astype(int)
    
#     # Fill NaN values
#     features_df = features_df.ffill().bfill().fillna(0)
    
#     print(f"✅ Created {len([col for col in features_df.columns if col not in ['timestamp', 'is_anomaly']])} features")
    
#     return features_df

# def evaluate_model_comprehensive(y_true, y_pred, model_name):
#     """Comprehensive model evaluation"""
#     print(f"\n📈 {model_name} Evaluation:")
#     print("=" * 50)
    
#     # Classification report
#     print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     print("Confusion Matrix:")
#     print(cm)
    
#     # Calculate detailed metrics
#     tn, fp, fn, tp = cm.ravel()
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
#     print(f"Precision: {precision:.3f}")
#     print(f"Recall: {recall:.3f}")
#     print(f"F1-Score: {f1:.3f}")
#     print(f"Specificity: {specificity:.3f}")
#     print(f"False Positive Rate: {false_positive_rate:.3f}")
    
#     return precision, recall, f1, specificity, false_positive_rate

# def plot_model_comparison(results):
#     """Plot comprehensive model comparison"""
#     models = list(results.keys())
#     metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     axes = axes.flatten()
    
#     for i, metric in enumerate(metrics):
#         values = [results[model][i] for model in models]
#         bars = axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
#         axes[i].set_title(f'{metric} Comparison', fontweight='bold')
#         axes[i].set_ylabel(metric)
#         axes[i].tick_params(axis='x', rotation=45)
#         axes[i].set_ylim(0, 1)
#         axes[i].grid(True, alpha=0.3)
        
#         # Add value labels
#         for bar, value in zip(bars, values):
#             axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
#                         f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     print("🚀 Starting Enhanced Robust Model Training")
#     print("Including BiLSTM for temporal pattern recognition")
#     print("=" * 60)
    
#     # Load the realistic dataset
#     df = pd.read_csv('training_dataset_realistic.csv', parse_dates=['timestamp'])
    
#     # Enhanced feature engineering
#     features_df = enhanced_feature_engineering(df)
    
#     # Prepare features
#     exclude_cols = ['timestamp', 'is_anomaly']
#     feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
#     X = features_df[feature_columns].values
#     y = features_df['is_anomaly'].values
    
#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.3, random_state=42, stratify=y
#     )
    
#     print(f"\n📊 Dataset Summary:")
#     print(f"   Total samples: {len(df)}")
#     print(f"   Anomalies: {sum(y)} ({sum(y)/len(y):.2%})")
#     print(f"   Training set: {X_train.shape[0]} samples")
#     print(f"   Test set: {X_test.shape[0]} samples")
#     print(f"   Features: {X_train.shape[1]}")
    
#     # Apply conservative SMOTE
#     print("\n🔄 Applying conservative SMOTE...")
#     smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Boost to 30% anomalies
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
#     print(f"   After SMOTE - Training: {X_train_balanced.shape[0]} samples")
#     print(f"   Anomaly rate: {sum(y_train_balanced)/len(y_train_balanced):.2%}")
    
#     # Train multiple models
#     models = {}
#     results = {}
    
#     print("\n🎯 Training Multiple Models...")
#     print("=" * 60)
    
#     # 1. Random Forest with Class Weighting
#     print("\n🌳 1. Random Forest (Class Weighting)")
#     rf_model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=15,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
    
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
    
#     results['Random Forest'] = evaluate_model_comprehensive(y_test, y_pred_rf, "Random Forest")
#     models['random_forest'] = rf_model
    
#     # 2. Random Forest with SMOTE
#     print("\n🌳 2. Random Forest (SMOTE)")
#     rf_smote_model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=15,
#         random_state=42,
#         n_jobs=-1
#     )
    
#     rf_smote_model.fit(X_train_balanced, y_train_balanced)
#     y_pred_rf_smote = rf_smote_model.predict(X_test)
    
#     results['Random Forest (SMOTE)'] = evaluate_model_comprehensive(y_test, y_pred_rf_smote, "Random Forest (SMOTE)")
#     models['random_forest_smote'] = rf_smote_model
    
#     # 3. Isolation Forest
#     print("\n🌲 3. Isolation Forest")
#     iso_model = IsolationForest(
#         n_estimators=100,
#         contamination=0.17,  # Based on actual anomaly rate
#         random_state=42,
#         n_jobs=-1
#     )
    
#     iso_model.fit(X_train)
#     y_pred_iso = iso_model.predict(X_test)
#     y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]
    
#     results['Isolation Forest'] = evaluate_model_comprehensive(y_test, y_pred_iso, "Isolation Forest")
#     models['isolation_forest'] = iso_model
    
#     # 4. BiLSTM Model
#     print("\n🧠 4. BiLSTM (Temporal Patterns)")
#     try:
#         bilstm_model, y_pred_bilstm, y_test_bilstm, history = train_bilstm(
#             X_train, y_train, X_test, y_test, feature_columns, sequence_length=10
#         )
        
#         results['BiLSTM'] = evaluate_model_comprehensive(y_test_bilstm, y_pred_bilstm, "BiLSTM")
#         models['bilstm'] = bilstm_model
        
#         # Plot BiLSTM predictions vs actual
#         plt.figure(figsize=(12, 6))
#         plt.plot(y_test_bilstm[:200], label='Actual', alpha=0.7)
#         plt.plot(y_pred_bilstm[:200], label='Predicted', alpha=0.7)
#         plt.title('BiLSTM Predictions vs Actual (First 200 samples)')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Anomaly (1) / Normal (0)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig('bilstm_predictions.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#     except Exception as e:
#         print(f"❌ BiLSTM training failed: {e}")
#         print("   Continuing with other models...")
    
#     # Compare all models
#     print("\n🏆 FINAL MODEL COMPARISON")
#     print("=" * 60)
    
#     comparison_df = pd.DataFrame(results, 
#                                 index=['Precision', 'Recall', 'F1-Score', 'Specificity', 'FPR']).T
#     comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
#     print(comparison_df)
    
#     # Plot comprehensive comparison
#     plot_model_comparison(results)
    
#     # Save the best model based on F1-Score
#     best_model_name = comparison_df.index[0].lower().replace(' ', '_').replace('(', '').replace(')', '')
#     best_model = models.get(best_model_name, models['random_forest'])
    
#     # Save models and components
#     joblib.dump(best_model, f'best_model_enhanced_{best_model_name}.pkl')
#     joblib.dump(scaler, 'scaler_enhanced.pkl')
#     joblib.dump(feature_columns, 'feature_names_enhanced.pkl')
    
#     print(f"\n🎉 ENHANCED TRAINING COMPLETED!")
#     print(f"🏆 Best Model: {comparison_df.index[0]}")
#     print(f"📊 Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.3f}")
#     print(f"🎯 Precision: {comparison_df.iloc[0]['Precision']:.3f}")
#     print(f"🔍 Recall: {comparison_df.iloc[0]['Recall']:.3f}")
#     print(f"💾 Saved: best_model_enhanced_{best_model_name}.pkl")
    
#     # Feature importance for tree-based models
#     if hasattr(best_model, 'feature_importances_'):
#         feature_importance = pd.DataFrame({
#             'feature': feature_columns,
#             'importance': best_model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         plt.figure(figsize=(12, 8))
#         top_features = feature_importance.head(15)
#         plt.barh(top_features['feature'], top_features['importance'])
#         plt.xlabel('Importance')
#         plt.title('Top 15 Feature Importance')
#         plt.gca().invert_yaxis()
#         plt.tight_layout()
#         plt.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
#         plt.show()
#
#
# PYTORCH ON CUDA

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class BiLSTMModel(nn.Module):
    """BiLSTM model for anomaly detection with PyTorch"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
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
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context_vector)
        return output

def create_sequences_pytorch(data, labels, sequence_length=10):
    """Create sequences for PyTorch training"""
    X_sequences = []
    y_sequences = []
    
    for i in range(len(data) - sequence_length):
        X_sequences.append(data[i:i + sequence_length])
        y_sequences.append(labels[i + sequence_length - 1])
    
    return np.array(X_sequences), np.array(y_sequences)

def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    """Train PyTorch model"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / train_total)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)
        
        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_bilstm_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
                  f"Val Acc: {val_accuracies[-1]:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_bilstm_model.pth'))
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def train_bilstm_pytorch(X_train, y_train, X_test, y_test, feature_names, sequence_length=10):
    """Train BiLSTM model with PyTorch"""
    print("🧠 Training BiLSTM Model (PyTorch)...")
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences_pytorch(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences_pytorch(X_test, y_test, sequence_length)
    
    print(f"   Training sequences: {X_train_seq.shape}")
    print(f"   Test sequences: {X_test_seq.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.LongTensor(y_train_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.LongTensor(y_test_seq)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = BiLSTMModel(
        input_size=X_train.shape[1],
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print("   Model architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train model
    model, train_losses, val_losses, train_acc, val_acc = train_pytorch_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=50
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        y_pred_proba = test_outputs.cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = np.mean(y_pred == y_test_seq)
    print(f"✅ BiLSTM trained with {accuracy:.2%} accuracy on test sequences")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bilstm_pytorch_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, y_pred, y_test_seq

def enhanced_feature_engineering(df):
    """Create robust features for all models"""
    features_df = df.copy()
    
    base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
    
    # Time-based features
    features_df['hour'] = df['timestamp'].dt.hour
    features_df['day_of_week'] = df['timestamp'].dt.dayofweek
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
    # Rolling statistics
    for feature in base_features:
        features_df[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
        features_df[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std()
        features_df[f'{feature}_rolling_min_5'] = df[feature].rolling(window=5, min_periods=1).min()
        features_df[f'{feature}_rolling_max_5'] = df[feature].rolling(window=5, min_periods=1).max()
        features_df[f'{feature}_roc'] = df[feature].pct_change().fillna(0)
    
    # Interaction features
    features_df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_percent'] + 1)
    features_df['system_load_index'] = (df['cpu_percent'] * 0.6 + df['memory_percent'] * 0.3 + df['disk_usage_percent'] * 0.1)
    features_df['resource_pressure'] = (df['cpu_percent'] > 80).astype(int) + (df['memory_percent'] > 80).astype(int) + (df['disk_usage_percent'] > 80).astype(int)
    
    # Fill NaN values
    features_df = features_df.ffill().bfill().fillna(0)
    
    print(f"✅ Created {len([col for col in features_df.columns if col not in ['timestamp', 'is_anomaly']])} features")
    
    return features_df

def evaluate_model_comprehensive(y_true, y_pred, model_name):
    """Comprehensive model evaluation"""
    print(f"\n📈 {model_name} Evaluation:")
    print("=" * 50)
    
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Specificity: {specificity:.3f}")
    
    return precision, recall, f1, specificity

def plot_model_comparison(results):
    """Plot comprehensive model comparison"""
    models = list(results.keys())
    metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][i] for model in models]
        bars = axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
        axes[i].set_title(f'{metric} Comparison', fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pytorch_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🚀 Starting Enhanced Model Training with PyTorch BiLSTM")
    print("=" * 60)
    
    # Load the realistic dataset
    df = pd.read_csv('training_dataset_realistic.csv', parse_dates=['timestamp'])
    
    # Enhanced feature engineering
    features_df = enhanced_feature_engineering(df)
    
    # Prepare features
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['is_anomaly'].values
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Anomalies: {sum(y)} ({sum(y)/len(y):.2%})")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Apply conservative SMOTE
    print("\n🔄 Applying conservative SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   After SMOTE - Training: {X_train_balanced.shape[0]} samples")
    print(f"   Anomaly rate: {sum(y_train_balanced)/len(y_train_balanced):.2%}")
    
    # Train multiple models
    models = {}
    results = {}
    
    print("\n🎯 Training Multiple Models...")
    print("=" * 60)
    
    # 1. Random Forest with Class Weighting
    print("\n🌳 1. Random Forest (Class Weighting)")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    results['Random Forest'] = evaluate_model_comprehensive(y_test, y_pred_rf, "Random Forest")
    models['random_forest'] = rf_model
    
    # 2. Random Forest with SMOTE
    print("\n🌳 2. Random Forest (SMOTE)")
    rf_smote_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_smote_model.fit(X_train_balanced, y_train_balanced)
    y_pred_rf_smote = rf_smote_model.predict(X_test)
    
    results['Random Forest (SMOTE)'] = evaluate_model_comprehensive(y_test, y_pred_rf_smote, "Random Forest (SMOTE)")
    models['random_forest_smote'] = rf_smote_model
    
    # 3. Isolation Forest
    print("\n🌲 3. Isolation Forest")
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.17,
        random_state=42,
        n_jobs=-1
    )
    
    iso_model.fit(X_train)
    y_pred_iso = iso_model.predict(X_test)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]
    
    results['Isolation Forest'] = evaluate_model_comprehensive(y_test, y_pred_iso, "Isolation Forest")
    models['isolation_forest'] = iso_model
    
    # 4. BiLSTM with PyTorch
    print("\n🧠 4. BiLSTM with PyTorch")
    try:
        bilstm_model, y_pred_bilstm, y_test_bilstm = train_bilstm_pytorch(
            X_train, y_train, X_test, y_test, feature_columns, sequence_length=10
        )
        
        results['BiLSTM (PyTorch)'] = evaluate_model_comprehensive(y_test_bilstm, y_pred_bilstm, "BiLSTM (PyTorch)")
        models['bilstm_pytorch'] = bilstm_model
        
    except Exception as e:
        print(f"❌ PyTorch BiLSTM training failed: {e}")
        print("   Make sure PyTorch is installed: pip install torch")
        print("   Continuing with other models...")
    
    # Compare all models
    print("\n🏆 FINAL MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame(results, 
                                index=['Precision', 'Recall', 'F1-Score', 'Specificity']).T
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print(comparison_df)
    
    # Plot comprehensive comparison
    plot_model_comparison(results)
    
    # Save the best model
    best_model_name = comparison_df.index[0].lower().replace(' ', '_').replace('(', '').replace(')', '')
    best_model = models.get(best_model_name, models['random_forest'])
    
    joblib.dump(best_model, f'best_model_pytorch_{best_model_name}.pkl')
    joblib.dump(scaler, 'scaler_pytorch.pkl')
    joblib.dump(feature_columns, 'feature_names_pytorch.pkl')
    
    print(f"\n🎉 PYTORCH TRAINING COMPLETED!")
    print(f"🏆 Best Model: {comparison_df.index[0]}")
    print(f"📊 Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.3f}")
    print(f"💾 Saved: best_model_pytorch_{best_model_name}.pkl")