# Phase 3: Real-Time Prediction

Welcome to **Phase 3: Real-Time Prediction** of the SystemSentinel System Failure Predictor project. This phase brings the anomaly detection models into a live environment, enabling real-time system health monitoring and alerting.

---

## Overview

In this phase, you will:
- Deploy trained models to make live predictions on incoming system data streams.
- Use specialized scripts for real-time anomaly detection using BiLSTM and other models.
- Automatically generate alerts for detected anomalies, enabling rapid response and system reliability.

---

## Directory Structure

- `real_time_bilstm_predictor.py`  
  Main script for applying a BiLSTM anomaly detection model in real time to system data streams.
- `real_time_bilstm_predictor_fixed.py`  
  An enhanced/fixed version of the real-time BiLSTM predictor with improvements or bug fixes.
- `alert_manager.py`  
  Handles alerting logic—sends notifications or triggers mitigation actions when anomalies are detected.

> For a full list of files, visit the [Phase-3-Real-Time-Prediction directory on GitHub](https://github.com/Villwin007/SystemSentinel-system-failure-predictor/tree/main/Phase-3-Real-Time-Prediction).

---

## Prerequisites

- Python 3.8+
- PyTorch and other dependencies (see project's `requirements.txt`)
- Trained models and scalers/artifacts from [Phase 2: Anomaly Detection](../Phase-2-Anomaly-Detection/)

Install dependencies:
```bash
pip install -r ../requirements.txt
```

---

## Usage Instructions

### 1. Real-Time Prediction with BiLSTM

To run the real-time BiLSTM predictor:
```bash
python real_time_bilstm_predictor.py
# or for the fixed/enhanced version:
python real_time_bilstm_predictor_fixed.py
```
- Configure input sources—these scripts typically read from system data streams, logs, or real-time feeds.
- The script will load the necessary trained model and preprocessing artifacts from Phase 2.

### 2. Alert Management

To enable alerting for detected anomalies:
```bash
python alert_manager.py
```
- Integrate with notification services (email, Slack, webhooks, etc.) as needed.
- Customize alert thresholds and response actions within the script.

---

## Contribution

Open source contributions are welcome! Please fork, branch, and submit pull requests. Document new features and ensure robust testing for all real-time functions.

---

## Next Steps

After validating your real-time pipeline, proceed to [Phase 4: Productionalization](../Phase-4-Productionalization/) for deployment and scaling.

---

© Villwin007. Licensed under the repository's open source license.
