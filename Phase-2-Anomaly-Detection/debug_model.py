import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def investigate_overfitting():
    print("🔍 INVESTIGATING MODEL PERFORMANCE")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('training_dataset_realistic.csv', parse_dates=['timestamp'])
    
    # Simple feature engineering
    features_df = df.copy()
    base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
    
    for feature in base_features:
        features_df[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
        features_df[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std()
        features_df[f'{feature}_roc'] = df[feature].pct_change().fillna(0)
    
    features_df['hour'] = df['timestamp'].dt.hour
    features_df = features_df.ffill().bfill().fillna(0)
    
    # Prepare features
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['is_anomaly'].values
    
    print(f"📊 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total anomalies: {sum(y)} ({sum(y)/len(y):.2%})")
    print(f"   Features: {len(feature_columns)}")
    
    # Check anomaly distribution
    anomaly_indices = np.where(y == 1)[0]
    print(f"\n📈 Anomaly Analysis:")
    print(f"   Anomaly indices: {anomaly_indices}")
    
    # Look at the actual anomaly values
    print(f"\n🔍 Examining Anomalous Records:")
    anomalies_df = features_df[features_df['is_anomaly'] == 1]
    print(anomalies_df[['cpu_percent', 'memory_percent', 'disk_usage_percent']].describe())
    
    # Check if anomalies are too obvious
    print(f"\n🎯 Are anomalies too obvious?")
    normal_stats = features_df[features_df['is_anomaly'] == 0][base_features].describe()
    anomaly_stats = features_df[features_df['is_anomaly'] == 1][base_features].describe()
    
    print("Normal data stats:")
    print(normal_stats.loc[['mean', 'max']])
    print("\nAnomaly data stats:")
    print(anomaly_stats.loc[['mean', 'max']])
    
    # Test different train/test splits
    print(f"\n🧪 Testing Different Splits:")
    
    splits = [0.2, 0.3, 0.4]
    for test_size in splits:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            max_depth=10
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"   Test size {test_size:.0%}: Precision={precision:.3f}, Recall={recall:.3f}")
        print(f"   Confusion Matrix:\n{cm}")
        
        if precision == 1.0:
            print(f"   ⚠️  100% precision - might be overfitting!")
    
    # Check feature importance
    print(f"\n📊 Feature Importance Analysis:")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Plot the actual anomalies vs normal points
    plt.figure(figsize=(15, 10))
    
    # Plot 1: CPU vs Memory
    plt.subplot(2, 3, 1)
    plt.scatter(features_df[features_df['is_anomaly'] == 0]['cpu_percent'],
                features_df[features_df['is_anomaly'] == 0]['memory_percent'],
                alpha=0.5, label='Normal', s=10)
    plt.scatter(features_df[features_df['is_anomaly'] == 1]['cpu_percent'],
                features_df[features_df['is_anomaly'] == 1]['memory_percent'],
                alpha=1.0, label='Anomaly', s=50, color='red', marker='x')
    plt.xlabel('CPU %')
    plt.ylabel('Memory %')
    plt.legend()
    plt.title('CPU vs Memory (Anomalies in Red)')
    
    # Plot 2: Distribution of key metrics
    for i, feature in enumerate(['cpu_percent', 'memory_percent', 'disk_usage_percent']):
        plt.subplot(2, 3, i+2)
        plt.hist(features_df[features_df['is_anomaly'] == 0][feature], 
                 alpha=0.7, label='Normal', bins=20, density=True)
        plt.hist(features_df[features_df['is_anomaly'] == 1][feature], 
                 alpha=0.7, label='Anomaly', bins=10, density=True, color='red')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('anomaly_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🔎 Conclusion:")
    if len(anomalies_df) < 20:
        print("   ❌ Too few anomalies for reliable training")
        print("   💡 Need to generate more realistic anomaly data")
    else:
        print("   ✅ Reasonable number of anomalies")
    
    # Check if anomalies are clearly separable
    anomaly_separation = False
    for feature in ['cpu_percent', 'memory_percent', 'disk_usage_percent']:
        normal_max = features_df[features_df['is_anomaly'] == 0][feature].max()
        anomaly_min = features_df[features_df['is_anomaly'] == 1][feature].min()
        if anomaly_min > normal_max * 1.5:  # If anomalies are much higher
            print(f"   ⚠️  Anomalies in {feature} are too obvious (easy to separate)")
            anomaly_separation = True
    
    if not anomaly_separation:
        print("   ✅ Anomalies appear realistically mixed with normal data")

if __name__ == '__main__':
    investigate_overfitting()