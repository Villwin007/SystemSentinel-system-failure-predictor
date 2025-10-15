# Phase 4: Productionalization

Welcome to **Phase 4: Productionalization** of the SystemSentinel System Failure Predictor project. This phase focuses on deploying the trained ML models and system monitoring logic into a production-ready, scalable, and user-friendly application. It includes backend APIs, a forecasting engine, and a modern frontend for visualization and control.

---

## Overview

In this phase, you will:
- Deploy the backend services (APIs and integration endpoints) using FastAPI.
- Integrate advanced forecasting and failure prediction modules.
- Set up a modern frontend for system monitoring and alerting.
- Ensure the entire stack is production-ready, scalable, and maintainable.

---

## Directory Structure

- `backend/`
  - [`fastapi_app.py`](./backend/fastapi_app.py): Main FastAPI backend serving ML inference and monitoring APIs.
  - [`fastapi_app_improved.py`](./backend/fastapi_app_improved.py): Enhanced backend with additional features, improved code, or optimizations.
- `forecasting/`
  - [`failure_predictor.py`](./forecasting/failure_predictor.py): Core system failure prediction logic for online inference.
  - [`resource_forecaster.py`](./forecasting/resource_forecaster.py): Forecasts system resource usage and trends.
  - [`integration_api.py`](./forecasting/integration_api.py): Exposes model predictions and forecasts via API endpoints for integration.
  - [`requirements.txt`](./forecasting/requirements.txt): Python dependencies for forecasting services.
- `frontend/`
  - [`index.html`](./frontend/index.html): Entry point for the web interface.
  - [`src/`](./frontend/src): Source code for frontend components and logic.
  - [`public/`](./frontend/public): Static assets.
  - [`package.json`](./frontend/package.json), [`package-lock.json`](./frontend/package-lock.json): NPM dependencies for the frontend.
  - [`vite.config.js`](./frontend/vite.config.js): Vite configuration for fast web development.
  - [`eslint.config.js`](./frontend/eslint.config.js): Linting configuration for code quality.
  - [`README.md`](./frontend/README.md): Detailed frontend usage and setup instructions.

> For a complete and up-to-date file listing, visit the [Phase-4-Productionalization directory on GitHub](https://github.com/Villwin007/SystemSentinel-system-failure-predictor/tree/main/Phase-4-Productionalization).

---

## Prerequisites

- Python 3.8+ (for backend and forecasting modules)
- Node.js 18+ and npm (for frontend)
- All dependencies listed in the [main requirements.txt](./requirements.txt) and submodule requirements files

Install backend dependencies:
```bash
pip install -r requirements.txt
pip install -r forecasting/requirements.txt
```

Install frontend dependencies:
```bash
cd frontend
npm install
```

---

## Usage Instructions

### 1. Start the Backend API

From the `backend` directory, run:
```bash
# Basic backend
python fastapi_app.py
# or improved backend
python fastapi_app_improved.py
```
- This starts the FastAPI server exposing ML inference and system monitoring endpoints.
- The backend will load models and forecasting logic from the `forecasting` submodule.

### 2. Run Forecasting Services

These scripts are run or imported as part of the backend, but can also be invoked directly for development:
```bash
python forecasting/failure_predictor.py
python forecasting/resource_forecaster.py
python forecasting/integration_api.py
```
- Ensure all models and required artifacts from earlier phases are accessible.

### 3. Launch the Frontend

From the `frontend` directory:
```bash
npm run dev
```
- This starts the Vite-powered development server.
- Open `http://localhost:5173` (or the printed port) in your browser.
- The frontend communicates with the backend APIs for live system status, predictions, and alerting.

### 4. Production Deployment

- For production, build the frontend:
  ```bash
  npm run build
  ```
  and serve the `dist/` directory using your preferred static site server.
- Deploy the backend and forecasting services using a WSGI server (e.g., Uvicorn, Gunicorn) and configure for scaling (Docker, systemd, etc.).

---

## Configuration

- Edit API endpoints and frontend URLs as necessary in configuration files or environment variables.
- Integrate authentication, logging, and monitoring as required for your operational environment.

---

## Contribution

Contributions are welcome! Please fork, branch, and submit pull requests. 
- Ensure code is well-documented and tested.
- Refer to the [`frontend/README.md`](./frontend/README.md) for web-specific contribution tips.

---

## Next Steps

Continue to monitor, optimize, and extend the platform in production. Consider adding advanced alerting, automated remediation, or integration with external monitoring tools.

---

© Villwin007. Licensed under the repository's open source license.
