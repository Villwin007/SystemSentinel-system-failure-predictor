# Phase 2: Anomaly Detection

Welcome to **Phase 2: Anomaly Detection** of the SystemSentinel System Failure Predictor project. This phase focuses on detecting anomalies in system data using advanced machine learning models and feature engineering techniques.

---

## Overview

During this phase, you will:
- Engineer features from system datasets.
- Train and evaluate various anomaly detection models (Random Forest, Isolation Forest, BiLSTM, etc.).
- Generate synthetic anomalies for robust model training.
- Compare models and analyze feature importance.
- Batch predict anomalies on new datasets.

---

## Directory Structure

- `feature_engineering.py`  
  Script for extracting and engineering features from raw/preprocessed data.
- `generate_dataset.py`  
  Generates datasets for model training and evaluation.
- `generate_better_anomalies.py`  
  Produces synthetic anomalies to improve model robustness.
- `train_model.py`, `train_model_enhanced.py`, `train_model_pytorch_enhanced.py`, `train_model_robust.py`  
  Scripts to train different anomaly detection models (classic, enhanced, PyTorch-based, robust).
- `batch_predict.py`  
  Perform batch predictions using trained models.
- `debug_model.py`  
  Debugging utilities for model development.
- Model & Preprocessing Artifacts:  
  - `all_models.pkl`, `best_model_enhanced_random_forest.pkl`, `best_model_isolation_forest.pkl`, `best_model_pytorch_random_forest.pkl`, `best_model_robust_random_forest_class_weight.pkl`, `best_bilstm_model.pth`  
  - `feature_names.pkl`, `feature_names_enhanced.pkl`, `feature_names_pytorch.pkl`, `feature_names_robust.pkl`
  - `scaler.pkl`, `scaler_enhanced.pkl`, `scaler_pytorch.pkl`, `scaler_robust.pkl`
- Data:
  - `training_dataset.csv`, `training_dataset_realistic.csv`
- Visualizations:
  - `anomaly_analysis.png`, `feature_selection_importance.png`, `model_comparison_enhanced.png`, `pytorch_model_comparison.png`, `robust_model_comparison.png`, `bilstm_pytorch_training_history.png`

> For a full list of files, visit the [Phase-2-Anomaly-Detection directory on GitHub](https://github.com/Villwin007/SystemSentinel-system-failure-predictor/tree/main/Phase-2-Anomaly-Detection).

---

## Prerequisites

- Python 3.8+
- PyTorch, scikit-learn, pandas, numpy, and other dependencies (see project's `requirements.txt`)
- Data from [Phase 1: Data Acquisition](../Phase-1-Data-Acquisition/)

Install dependencies:
```bash
pip install -r ../requirements.txt
```

---

## Usage Instructions

### 1. Feature Engineering

To generate features from your dataset:
```bash
python feature_engineering.py
```
Customize input and output file paths as needed within the script.

### 2. Generate Synthetic Dataset & Anomalies

To create a training dataset or more realistic anomalies:
```bash
python generate_dataset.py
python generate_better_anomalies.py
```

### 3. Model Training

You can train different models:
- **Classic/Basic Model:**  
  ```bash
  python train_model.py
  ```
- **Enhanced Random Forest:**  
  ```bash
  python train_model_enhanced.py
  ```
- **PyTorch-based Models:**  
  ```bash
  python train_model_pytorch_enhanced.py
  ```
- **Robust Model (Class Weights):**  
  ```bash
  python train_model_robust.py
  ```

Model artifacts (`.pkl`, `.pth`) will be saved for later use.

### 4. Batch Prediction

To predict anomalies on a batch of new data:
```bash
python batch_predict.py
```
Edit script or pass parameters for model and data paths as needed.

### 5. Debugging & Visualization

- Use `debug_model.py` to troubleshoot or fine-tune models.
- Inspect `.png` visualization files for model comparison, feature importance, and training history.

---

## Contribution

Open source contributions are encouraged! Please fork, branch, and submit pull requests. Ensure new scripts are well-documented and tested.

---

## Next Steps

After anomaly detection, proceed to [Phase 3: Real-Time Prediction](../Phase-3-Real-Time-Prediction/).

---

© Villwin007. Licensed under the repository's open source license.
