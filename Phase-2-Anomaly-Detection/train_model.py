import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from feature_engineering import create_features, prepare_training_data
import warnings
warnings.filterwarnings('ignore')

def train_isolation_forest(X_train, y_train, contamination=0.01):
    """Train Isolation Forest model"""
    print("🌲 Training Isolation Forest...")
    
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,  # Expected anomaly rate
        random_state=42,
        n_jobs=-1
    )
    
    # Isolation Forest uses fit_predict and returns -1 for anomalies, 1 for normal
    # We need to convert to our labeling (0=normal, 1=anomaly)
    y_pred = model.fit_predict(X_train)
    y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to our format
    
    # Calculate accuracy on known anomalies (from our synthetic data)
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ Isolation Forest trained with {accuracy:.2%} accuracy on training data")
    
    return model, y_pred

def train_one_class_svm(X_train, y_train, nu=0.01):
    """Train One-Class SVM model"""
    print("🔍 Training One-Class SVM...")
    
    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=nu  # Upper bound on anomaly fraction
        # Note: OneClassSVM doesn't have random_state parameter
    )
    
    y_pred = model.fit_predict(X_train)
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ One-Class SVM trained with {accuracy:.2%} accuracy on training data")
    
    return model, y_pred

def train_local_outlier_factor(X_train, y_train, contamination=0.01):
    """Train Local Outlier Factor model"""
    print("📊 Training Local Outlier Factor...")
    
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
        novelty=True  # Important for prediction on new data
    )
    
    y_pred = model.fit_predict(X_train)
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    
    accuracy = np.mean(y_pred == y_train)
    print(f"✅ LOF trained with {accuracy:.2%} accuracy on training data")
    
    return model, y_pred

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    print(f"\n📈 {model_name} Evaluation:")
    print("=" * 50)
    
    # Classification report
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return precision, recall, f1

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for interpretable models"""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f"Feature Importance - {model_name}", fontsize=14, fontweight='bold')
        bars = plt.bar(range(len(importances)), importances[indices], color='skyblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance Score', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances[indices]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{importance:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top 10 features
        print(f"\n🔝 Top 10 Important Features for {model_name}:")
        for i in range(min(10, len(importances))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def compare_models(X_train, y_train, feature_names):
    """Train and compare multiple models"""
    models = {}
    results = {}
    
    # Train different models
    print("🚀 Training Multiple Anomaly Detection Models...")
    print("=" * 60)
    
    try:
        # Isolation Forest
        if_model, if_pred = train_isolation_forest(X_train, y_train, contamination=0.01)
        models['isolation_forest'] = if_model
        results['isolation_forest'] = evaluate_model(y_train, if_pred, "Isolation Forest")
    except Exception as e:
        print(f"❌ Error training Isolation Forest: {e}")
    
    try:
        # One-Class SVM
        svm_model, svm_pred = train_one_class_svm(X_train, y_train, nu=0.01)
        models['one_class_svm'] = svm_model
        results['one_class_svm'] = evaluate_model(y_train, svm_pred, "One-Class SVM")
    except Exception as e:
        print(f"❌ Error training One-Class SVM: {e}")
    
    try:
        # Local Outlier Factor
        lof_model, lof_pred = train_local_outlier_factor(X_train, y_train, contamination=0.01)
        models['local_outlier_factor'] = lof_model
        results['local_outlier_factor'] = evaluate_model(y_train, lof_pred, "Local Outlier Factor")
    except Exception as e:
        print(f"❌ Error training Local Outlier Factor: {e}")
    
    # Compare results
    print("\n🏆 Model Comparison:")
    print("=" * 50)
    if results:
        comparison_df = pd.DataFrame(results, index=['Precision', 'Recall', 'F1-Score']).T
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print(comparison_df)
        
        # Plot feature importance for the best model
        best_model_name = comparison_df.index[0]
        best_model = models[best_model_name]
        
        if best_model_name == 'isolation_forest':
            plot_feature_importance(best_model, feature_names, "Isolation Forest")
        
        return models, comparison_df, best_model_name
    else:
        print("❌ No models were successfully trained")
        return {}, pd.DataFrame(), ""

def save_models(models, scaler, feature_names, best_model_name):
    """Save trained models and metadata"""
    if not models:
        print("❌ No models to save")
        return
    
    # Save the best model
    if best_model_name:
        joblib.dump(models[best_model_name], f'best_model_{best_model_name}.pkl')
        print(f"💾 Saved best model: best_model_{best_model_name}.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("💾 Saved feature scaler: scaler.pkl")
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    print("💾 Saved feature names: feature_names.pkl")
    
    # Save all models
    joblib.dump(models, 'all_models.pkl')
    print("💾 Saved all models: all_models.pkl")

if __name__ == '__main__':
    print("🚀 Starting Model Training...")
    
    # Load and prepare data
    df = pd.read_csv('training_dataset.csv', parse_dates=['timestamp'])
    features_df = create_features(df)
    X, y, scaler, feature_names = prepare_training_data(features_df)
    
    # Split data (use all for training since we have few anomalies)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"🎯 Anomaly rate in training: {y_train.mean():.2%}")
    print(f"🎯 Anomaly rate in test: {y_test.mean():.2%}")
    
    # Train and compare models
    models, results, best_model_name = compare_models(X_train, y_train, feature_names)
    
    # Save models
    save_models(models, scaler, feature_names, best_model_name)
    
    if best_model_name:
        print(f"\n🎉 Training completed! Best model: {best_model_name}")
        print("📁 Files created:")
        print("   - best_model_*.pkl (The best performing model)")
        print("   - all_models.pkl (All trained models)")
        print("   - scaler.pkl (Feature scaler for new data)")
        print("   - feature_names.pkl (List of feature names)")
        
        # Show performance summary
        print(f"\n📊 Best Model Performance Summary:")
        best_result = results.loc[best_model_name]
        print(f"   Precision: {best_result['Precision']:.3f}")
        print(f"   Recall: {best_result['Recall']:.3f}")
        print(f"   F1-Score: {best_result['F1-Score']:.3f}")
    else:
        print("\n❌ Training failed. Please check the error messages above.")