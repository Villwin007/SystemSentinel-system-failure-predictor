# Phase 1: Data Acquisition

Welcome to **Phase 1: Data Acquisition** of the SystemSentinel System Failure Predictor project. This phase focuses on collecting and ingesting raw system data, laying the foundation for subsequent anomaly detection and prediction phases.

## Overview

In this phase, we provide tools and scripts to:
- Collect system-level data from various sources (e.g., logs, metrics, sensors).
- Ingest and preprocess the collected data for downstream analysis.
- Run the acquisition pipeline in a reproducible, containerized environment.

## Directory Structure

- `data_collector.py`  
  Collects raw system data from configured sources.
- `data_ingestor.py`  
  Handles ingesting, formatting, and storing collected data.
- `docker-compose.yml`  
  Provides a containerized environment for data collection and ingestion.

## Prerequisites

- Python 3.8+
- Docker & Docker Compose

Install required Python packages:
```bash
pip install -r ../requirements.txt
```

## Usage Instructions

### 1. Running Data Collection

To start collecting data, run:
```bash
python data_collector.py
```
Edit `data_collector.py` as needed to configure data sources, such as system logs or sensor endpoints.

### 2. Ingesting Collected Data

Once data is collected, ingest and preprocess it using:
```bash
python data_ingestor.py
```
You may configure input/output paths and data formatting parameters within the script.

### 3. Using Docker Compose

For a fully reproducible setup, use Docker Compose:
```bash
docker-compose up
```
This will spin up all necessary services as defined in `docker-compose.yml`, ensuring isolated and repeatable data acquisition runs.

## Contributing

Open source contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. Ensure your code is well-documented and tested.

## Next Steps

Proceed to [Phase 2: Anomaly Detection](../Phase-2-Anomaly-Detection/) after acquiring and ingesting your data.

---

© Villwin007. Licensed under the chosen open source license.  
