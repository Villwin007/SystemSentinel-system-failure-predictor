import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

def robust_feature_engineering(df):
    """Create features without over-engineering"""
    features_df = df.copy()
    
    # Basic features only - avoid creating too many synthetic-looking features
    base_features = ['cpu_percent', 'memory_percent', 'memory_used_gb', 'disk_usage_percent', 'disk_used_gb']
    
    # Conservative rolling statistics
    for feature in base_features:
        # Simple rolling features
        features_df[f'{feature}_rolling_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
        features_df[f'{feature}_rolling_std_5'] = df[feature].rolling(window=5, min_periods=1).std()
        
        # Rate of change
        features_df[f'{feature}_roc'] = df[feature].pct_change().fillna(0)
        
        # Simple z-score
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        features_df[f'{feature}_zscore'] = (df[feature] - mean_val) / std_val if std_val > 0 else 0
    
    # Time features
    features_df['hour'] = df['timestamp'].dt.hour
    features_df['day_of_week'] = df['timestamp'].dt.dayofweek
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    
    # Simple interaction features
    features_df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_percent'] + 1)
    features_df['system_load'] = (df['cpu_percent'] + df['memory_percent']) / 2
    
    # Fill NaN values conservatively
    features_df = features_df.ffill().bfill().fillna(0)
    
    print(f"✅ Created {len([col for col in features_df.columns if col not in ['timestamp', 'is_anomaly']])} robust features")
    
    return features_df

def conservative_smote(X, y, k_neighbors=2):
    """Use SMOTE conservatively to avoid overfitting"""
    print("🔄 Applying conservative SMOTE...")
    
    # Only create a modest number of synthetic samples
    smote = SMOTE(
        random_state=42, 
        k_neighbors=min(k_neighbors, sum(y == 1) - 1),  # Very conservative
        sampling_strategy=0.1  # Only boost to 10% anomalies (not 50%)
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"📊 Before: {X.shape[0]} samples, {sum(y == 1)} anomalies ({sum(y == 1)/len(y):.2%})")
    print(f"📊 After: {X_resampled.shape[0]} samples, {sum(y_resampled == 1)} anomalies ({sum(y_resampled == 1)/len(y_resampled):.2%})")
    
    return X_resampled, y_resampled

def train_with_cross_validation(model, X, y, model_name, cv_folds=5):
    """Train model with proper cross-validation"""
    print(f"🔍 {model_name} Cross-Validation...")
    
    # Use stratified k-fold for imbalanced data
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    print(f"   Cross-val F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores.mean(), cv_scores.std()

def evaluate_on_original_test(X_train, X_test, y_train, y_test, model, model_name):
    """Evaluate on original (non-synthetic) test data"""
    print(f"🧪 {model_name} Test Evaluation (Original Data):")
    
    # Train on balanced data, test on original imbalanced data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}") 
    print(f"   F1-Score: {f1:.3f}")
    print(f"   Confusion Matrix:\n{cm}")
    
    return precision, recall, f1, cm

def plot_learning_curve(train_sizes, train_scores, test_scores, model_name):
    """Plot learning curve to check for overfitting"""
    plt.figure(figsize=(10, 6))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'learning_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🚀 Starting Robust Model Training (Anti-Overfitting)")
    print("Goal: Achieve 75-85% precision WITHOUT overfitting")
    
    # Load data
    df = pd.read_csv('training_dataset.csv', parse_dates=['timestamp'])
    
    # Robust feature engineering (avoid over-engineering)
    features_df = robust_feature_engineering(df)
    
    # Prepare features
    exclude_cols = ['timestamp', 'is_anomaly']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns].values
    y = features_df['is_anomaly'].values
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data - keep original test set completely separate
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y  # Larger test set
    )
    
    print(f"\n📊 Data Summary:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples") 
    print(f"   Anomaly rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Apply CONSERVATIVE SMOTE only to training data
    X_train_balanced, y_train_balanced = conservative_smote(X_train, y_train)
    
    # Train models with proper validation
    models = {}
    results = {}
    
    print("\n🎯 Training Models with Robust Validation...")
    print("=" * 60)
    
    # 1. Random Forest with class weights (no SMOTE)
    print("\n🌳 1. Random Forest (Class Weighting - No SMOTE)")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # Handle imbalance internally
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation on original data
    rf_cv_mean, rf_cv_std = train_with_cross_validation(rf_model, X_train, y_train, "Random Forest")
    
    # Train and test
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn, fp, fn, tp = cm_rf.ravel()
    precision_rf = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_rf = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf) if (precision_rf + recall_rf) > 0 else 0
    
    models['random_forest_class_weight'] = rf_model
    results['random_forest_class_weight'] = (precision_rf, recall_rf, f1_rf)
    
    print(f"   Test Precision: {precision_rf:.3f}")
    print(f"   Test Recall: {recall_rf:.3f}")
    print(f"   Test F1: {f1_rf:.3f}")
    
    # 2. Random Forest with conservative SMOTE
    print("\n🌳 2. Random Forest (Conservative SMOTE)")
    rf_smote_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_smote_model.fit(X_train_balanced, y_train_balanced)
    y_pred_rf_smote = rf_smote_model.predict(X_test)  # Test on ORIGINAL test data
    
    cm_rf_smote = confusion_matrix(y_test, y_pred_rf_smote)
    tn, fp, fn, tp = cm_rf_smote.ravel()
    precision_rf_smote = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_rf_smote = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_rf_smote = 2 * (precision_rf_smote * recall_rf_smote) / (precision_rf_smote + recall_rf_smote) if (precision_rf_smote + recall_rf_smote) > 0 else 0
    
    models['random_forest_smote'] = rf_smote_model
    results['random_forest_smote'] = (precision_rf_smote, recall_rf_smote, f1_rf_smote)
    
    print(f"   Test Precision: {precision_rf_smote:.3f}")
    print(f"   Test Recall: {recall_rf_smote:.3f}")
    print(f"   Test F1: {f1_rf_smote:.3f}")
    
    # 3. Isolation Forest (unsupervised - no label issues)
    print("\n🌲 3. Isolation Forest (Unsupervised)")
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.01,  # Based on actual anomaly rate
        random_state=42,
        n_jobs=-1
    )
    
    iso_model.fit(X_train)  # No labels needed
    y_pred_iso = iso_model.predict(X_test)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]  # Convert to our format
    
    cm_iso = confusion_matrix(y_test, y_pred_iso)
    tn, fp, fn, tp = cm_iso.ravel()
    precision_iso = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_iso = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_iso = 2 * (precision_iso * recall_iso) / (precision_iso + recall_iso) if (precision_iso + recall_iso) > 0 else 0
    
    models['isolation_forest'] = iso_model
    results['isolation_forest'] = (precision_iso, recall_iso, f1_iso)
    
    print(f"   Test Precision: {precision_iso:.3f}")
    print(f"   Test Recall: {recall_iso:.3f}")
    print(f"   Test F1: {f1_iso:.3f}")
    
    # Compare results
    print("\n🏆 Robust Model Comparison (Test Set Performance):")
    print("=" * 60)
    comparison_df = pd.DataFrame(results, index=['Precision', 'Recall', 'F1-Score']).T
    comparison_df = comparison_df.sort_values('Precision', ascending=False)
    print(comparison_df)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    x_pos = np.arange(len(comparison_df))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = comparison_df[metric].values
        bars = plt.bar(x_pos, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title(f'{metric} Comparison')
        plt.xticks(x_pos, comparison_df.index, rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('robust_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the best model based on precision
    best_model_name = comparison_df.index[0]
    best_model = models[best_model_name]
    
    joblib.dump(best_model, f'best_model_robust_{best_model_name}.pkl')
    joblib.dump(scaler, 'scaler_robust.pkl')
    joblib.dump(feature_columns, 'feature_names_robust.pkl')
    
    print(f"\n🎉 Robust training completed!")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📈 Best precision: {comparison_df.loc[best_model_name, 'Precision']:.3f}")
    print(f"💾 Saved: best_model_robust_{best_model_name}.pkl")
    
    # Final reality check
    print(f"\n🔍 Reality Check:")
    print(f"   Expected precision range: 70-85% (realistic)")
    print(f"   If precision > 90%, be suspicious of overfitting")
    print(f"   If precision < 60%, need more feature engineering")