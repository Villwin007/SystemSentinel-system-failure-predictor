import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

def enhanced_feature_engineering(df):
    """Create advanced time-series features"""
    features_df = df.copy()
    
    # Basic features
    base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
    
    # Enhanced rolling statistics with multiple windows
    windows = [5, 10, 20]  # Multiple time windows
    for feature in base_features:
        for window in windows:
            # Rolling statistics
            features_df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
            features_df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window, min_periods=1).std()
            features_df[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window=window, min_periods=1).min()
            features_df[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window=window, min_periods=1).max()
            
            # Rolling percentiles
            features_df[f'{feature}_rolling_25p_{window}'] = df[feature].rolling(window=window, min_periods=1).quantile(0.25)
            features_df[f'{feature}_rolling_75p_{window}'] = df[feature].rolling(window=window, min_periods=1).quantile(0.75)
        
        # Enhanced rate of change and acceleration
        features_df[f'{feature}_roc_1'] = df[feature].pct_change(periods=1).fillna(0)
        features_df[f'{feature}_roc_5'] = df[feature].pct_change(periods=5).fillna(0)
        features_df[f'{feature}_acceleration'] = features_df[f'{feature}_roc_1'].diff().fillna(0)
        
        # Volatility measures
        features_df[f'{feature}_volatility'] = df[feature].rolling(window=10, min_periods=1).std() / df[feature].rolling(window=10, min_periods=1).mean()
        features_df[f'{feature}_volatility'] = features_df[f'{feature}_volatility'].fillna(0)
    
    # Cross-feature interactions
    features_df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_percent'] + 0.001)  # Avoid division by zero
    features_df['memory_disk_ratio'] = df['memory_percent'] / (df['disk_usage_percent'] + 0.001)
    features_df['system_load_index'] = (df['cpu_percent'] * 0.6 + df['memory_percent'] * 0.3 + df['disk_usage_percent'] * 0.1)
    
    # Trend features
    for feature in base_features:
        features_df[f'{feature}_trend'] = df[feature].rolling(window=10, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )
    
    # Time-based features
    features_df['hour'] = df['timestamp'].dt.hour
    features_df['day_of_week'] = df['timestamp'].dt.dayofweek
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
    # Anomaly-specific features (features that might indicate problems)
    features_df['high_cpu_low_memory'] = ((df['cpu_percent'] > 80) & (df['memory_percent'] < 20)).astype(int)
    features_df['memory_pressure'] = (df['memory_percent'] > 90).astype(int)
    features_df['disk_pressure'] = (df['disk_usage_percent'] > 90).astype(int)
    
    # Fill NaN values
    features_df = features_df.ffill().bfill().fillna(0)
    
    print(f"✅ Created {len([col for col in features_df.columns if col not in ['timestamp', 'is_anomaly']])} advanced features")
    
    return features_df

def handle_imbalanced_data(X, y, method='smote'):
    """Handle imbalanced dataset"""
    print(f"🔄 Handling class imbalance using {method.upper()}...")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=min(5, sum(y == 1) - 1))
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=42)
    else:
        return X, y  # No sampling
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    print(f"📊 Before: {X.shape[0]} samples, {sum(y == 1)} anomalies ({sum(y == 1)/len(y):.2%})")
    print(f"📊 After: {X_resampled.shape[0]} samples, {sum(y_resampled == 1)} anomalies ({sum(y_resampled == 1)/len(y_resampled):.2%})")
    
    return X_resampled, y_resampled

def train_optimized_isolation_forest(X_train, y_train):
    """Train optimized Isolation Forest with hyperparameter tuning"""
    print("🌲 Training Optimized Isolation Forest...")
    
    # Parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': ['auto', 0.5, 0.8],
        'contamination': [0.005, 0.01, 0.02],
        'max_features': [0.5, 0.8, 1.0]
    }
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        IsolationForest(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Convert predictions to our format
    y_pred = best_model.predict(X_train)
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ Optimized Isolation Forest trained with {accuracy:.2%} accuracy")
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    
    return best_model, y_pred

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier (supervised approach)"""
    print("🌳 Training Random Forest Classifier...")
    
    # Parameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_train)
    
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ Random Forest trained with {accuracy:.2%} accuracy")
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    
    return best_model, y_pred

def train_ensemble_model(X_train, y_train):
    """Train ensemble of multiple models"""
    print("🤝 Training Ensemble Model...")
    
    from sklearn.ensemble import VotingClassifier
    
    # Define individual models
    models = [
        ('isolation_forest', IsolationForest(
            n_estimators=100, 
            contamination=0.01, 
            random_state=42
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ))
    ]
    
    # Create ensemble (we need to convert Isolation Forest outputs)
    class EnsembleAnomalyDetector:
        def __init__(self, models):
            self.models = models
            
        def fit(self, X, y):
            for name, model in self.models:
                if hasattr(model, 'fit'):
                    model.fit(X, y if name != 'isolation_forest' else X)
            return self
            
        def predict(self, X):
            predictions = []
            for name, model in self.models:
                if name == 'isolation_forest':
                    pred = model.predict(X)
                    pred = [1 if x == -1 else 0 for x in pred]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            # Majority voting
            ensemble_pred = np.round(np.mean(predictions, axis=0))
            return ensemble_pred.astype(int)
    
    ensemble = EnsembleAnomalyDetector(models)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_train)
    
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ Ensemble model trained with {accuracy:.2%} accuracy")
    
    return ensemble, y_pred

def evaluate_model_enhanced(y_true, y_pred, model_name, X_test=None, y_test=None, model=None):
    """Enhanced model evaluation with more metrics"""
    print(f"\n📈 {model_name} Evaluation:")
    print("=" * 60)
    
    # Classification report
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"False Positive Rate: {fp/(fp+tn):.3f}")
    
    # Cross-validation scores if test data available
    if X_test is not None and y_test is not None and model is not None:
        try:
            # For supervised models
            if hasattr(model, 'predict_proba'):
                cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
                print(f"Cross-validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except:
            pass
    
    return precision, recall, f1, specificity

def plot_metrics_comparison(results):
    """Plot comparison of all models"""
    models = list(results.keys())
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model][i] for model in models]
        axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_selection(X, y, feature_names, method='random_forest'):
    """Select most important features"""
    print(f"🔍 Performing feature selection using {method}...")
    
    if method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top features (keep at least 15)
        n_features = max(15, len(feature_names) // 3)
        selected_indices = indices[:n_features]
        
        print(f"📊 Selected top {n_features} features from {len(feature_names)} total features")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance for Selection", fontsize=14, fontweight='bold')
        bars = plt.bar(range(n_features), importances[selected_indices][:n_features])
        plt.xticks(range(n_features), [feature_names[i] for i in selected_indices[:n_features]], 
                  rotation=45, ha='right')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_selection_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return X[:, selected_indices], [feature_names[i] for i in selected_indices]
    
    return X, feature_names

if __name__ == '__main__':
    print("🚀 Starting Enhanced Model Training...")
    print("Goal: Improve precision to 75-85% range")
    
    # Load data
    df = pd.read_csv('training_dataset.csv', parse_dates=['timestamp'])
    
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
    
    # Feature selection
    X_selected, selected_features = feature_selection(X_scaled, y, feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Summary:")
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Anomaly rate: {y_train.mean():.2%}")
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_imbalanced_data(X_train, y_train, method='smote')
    
    # Train multiple enhanced models
    models = {}
    results = {}
    
    print("\n🎯 Training Enhanced Models...")
    print("=" * 60)
    
    # 1. Optimized Isolation Forest
    if_model, if_pred = train_optimized_isolation_forest(X_train, y_train)
    models['optimized_isolation_forest'] = if_model
    results['optimized_isolation_forest'] = evaluate_model_enhanced(y_train, if_pred, "Optimized Isolation Forest")
    
    # 2. Random Forest (Supervised)
    rf_model, rf_pred = train_random_forest(X_train_balanced, y_train_balanced)
    models['random_forest'] = rf_model
    results['random_forest'] = evaluate_model_enhanced(y_train_balanced, rf_pred, "Random Forest", X_test, y_test, rf_model)
    
    # 3. Ensemble Model
    ensemble_model, ensemble_pred = train_ensemble_model(X_train_balanced, y_train_balanced)
    models['ensemble'] = ensemble_model
    results['ensemble'] = evaluate_model_enhanced(y_train_balanced, ensemble_pred, "Ensemble Model")
    
    # Compare results
    print("\n🏆 Enhanced Model Comparison:")
    print("=" * 60)
    comparison_df = pd.DataFrame(results, index=['Precision', 'Recall', 'F1-Score', 'Specificity']).T
    comparison_df = comparison_df.sort_values('Precision', ascending=False)
    print(comparison_df)
    
    # Plot comparison
    plot_metrics_comparison(results)
    
    # Save best model based on precision
    best_model_name = comparison_df.index[0]
    best_model = models[best_model_name]
    
    joblib.dump(best_model, f'best_model_enhanced_{best_model_name}.pkl')
    joblib.dump(scaler, 'scaler_enhanced.pkl')
    joblib.dump(selected_features, 'feature_names_enhanced.pkl')
    
    print(f"\n🎉 Enhanced training completed!")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📈 Best precision: {comparison_df.loc[best_model_name, 'Precision']:.3f}")
    print(f"💾 Saved: best_model_enhanced_{best_model_name}.pkl")
    
    # Test on holdout set
    if best_model_name != 'optimized_isolation_forest':  # Isolation Forest needs different prediction
        y_test_pred = best_model.predict(X_test)
        test_precision, test_recall, test_f1, _ = evaluate_model_enhanced(y_test, y_test_pred, f"{best_model_name} - Test Set")
        print(f"\n🧪 Test Set Performance:")
        print(f"   Precision: {test_precision:.3f}")
        print(f"   Recall: {test_recall:.3f}")
        print(f"   F1-Score: {test_f1:.3f}")