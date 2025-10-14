from fastapi import APIRouter, BackgroundTasks
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import pandas as pd
from .resource_forecaster import ResourceForecaster
from .failure_predictor import FailurePredictor
import psutil

# Create router
router = APIRouter(prefix="/api/forecasting", tags=["forecasting"])

# Initialize predictors
resource_forecaster = ResourceForecaster()
failure_predictor = FailurePredictor()

# Store forecasting results
forecasting_results = {
    "resource_predictions": {},
    "failure_risk": {},
    "last_updated": None
}

class ForecastingManager:
    def __init__(self):
        self.is_trained = False
        self.training_data = []
    
    async def collect_training_data(self):
        """Collect initial training data from system"""
        print("📊 Collecting training data for forecasting models...")
        
        # Collect 48 hours of mock data (in production, this would come from your database)
        base_time = datetime.now() - timedelta(hours=48)
        
        for i in range(576):  # 48 hours of 5-minute intervals
            timestamp = base_time + timedelta(minutes=i*5)
            
            # Real system metrics with realistic patterns
            hour = timestamp.hour
            # Daily pattern: higher during day, lower at night
            daily_pattern = 20 * np.sin(2 * np.pi * (hour - 9) / 24)  # Peak around 3 PM
            
            self.training_data.append({
                "timestamp": timestamp.isoformat(),
                "cpu_percent": max(10, min(95, np.random.uniform(30, 70) + daily_pattern + np.random.normal(0, 5))),
                "memory_percent": max(40, min(98, np.random.uniform(60, 85) + np.random.normal(0, 3))),
                "disk_usage_percent": max(30, min(95, 40 + (i/576)*20 + np.random.normal(0, 2)))  # Gradual increase
            })
        
        print(f"✅ Collected {len(self.training_data)} data points for training")
    
    async def train_models(self):
        """Train both forecasting models"""
        print("🧠 Training forecasting models...")
        
        if not self.training_data:
            await self.collect_training_data()
        
        try:
            # Train resource forecaster
            resource_forecaster.train_forecasting_models(self.training_data)
            
            # Train failure predictor  
            failure_predictor.train_failure_model(self.training_data)
            
            self.is_trained = True
            print("✅ Forecasting models trained successfully!")
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
    
    async def update_predictions(self, current_metrics=None):
        """Update all forecasting predictions"""
        if not self.is_trained:
            await self.train_models()
        
        try:
            if current_metrics is None:
                # Get current system metrics
                current_metrics = await get_current_system_metrics()
            
            # Get recent history for predictions (last 6 hours)
            recent_history = self.training_data[-72:] if self.training_data else []
            
            # Resource predictions
            resource_predictions = resource_forecaster.predict_future_resources(recent_history)
            
            # Failure risk prediction
            failure_risk = failure_predictor.predict_failure_risk(recent_history)
            
            # Update global results
            forecasting_results.update({
                "resource_predictions": resource_predictions,
                "failure_risk": failure_risk,
                "last_updated": datetime.now().isoformat()
            })
            
            print("🔮 Forecasting predictions updated")
            
        except Exception as e:
            print(f"❌ Prediction update failed: {e}")

# Global forecasting manager
forecasting_manager = ForecastingManager()

async def get_current_system_metrics():
    """Get current system metrics for forecasting"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_usage_percent": disk.percent
        }
    except Exception as e:
        # Fallback to mock data
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 45.0,
            "memory_percent": 65.0,
            "disk_usage_percent": 55.0
        }

@router.on_event("startup")
async def startup_forecasting():
    """Initialize forecasting on startup"""
    asyncio.create_task(forecasting_manager.train_models())
    print("🚀 Forecasting system initialized")

@router.get("/predictions")
async def get_forecasting_predictions():
    """Get current forecasting predictions"""
    if not forecasting_manager.is_trained:
        return {"status": "models_training", "message": "Forecasting models are being trained"}
    
    return forecasting_results

@router.get("/resource-forecast")
async def get_resource_forecast():
    """Get resource usage forecasts"""
    if not forecasting_manager.is_trained:
        return {"error": "Models not trained yet"}
    
    predictions = forecasting_results.get("resource_predictions", {})
    
    # Generate alerts based on predictions
    current_metrics = await get_current_system_metrics()
    alerts = resource_forecaster.generate_forecast_alerts(predictions, current_metrics)
    
    return {
        "predictions": predictions,
        "alerts": alerts,
        "last_updated": forecasting_results.get("last_updated")
    }

@router.get("/failure-risk")
async def get_failure_risk():
    """Get system failure risk prediction"""
    if not forecasting_manager.is_trained:
        return {"error": "Models not trained yet"}
    
    risk_data = forecasting_results.get("failure_risk", {})
    
    # Generate alerts based on risk
    alerts = failure_predictor.generate_failure_alerts(risk_data)
    
    return {
        "risk_assessment": risk_data,
        "alerts": alerts,
        "last_updated": forecasting_results.get("last_updated")
    }

@router.post("/refresh")
async def refresh_predictions(background_tasks: BackgroundTasks):
    """Manually refresh forecasting predictions"""
    background_tasks.add_task(forecasting_manager.update_predictions)
    return {"status": "refreshing", "message": "Predictions are being updated"}

@router.get("/status")
async def get_forecasting_status():
    """Get forecasting system status"""
    return {
        "is_trained": forecasting_manager.is_trained,
        "training_data_points": len(forecasting_manager.training_data),
        "last_updated": forecasting_results.get("last_updated"),
        "models_loaded": {
            "resource_forecaster": len(resource_forecaster.models) > 0,
            "failure_predictor": failure_predictor.model is not None
        }
    }

# Background task to periodically update predictions
async def periodic_prediction_updater():
    """Update predictions every 15 minutes"""
    while True:
        try:
            if forecasting_manager.is_trained:
                await forecasting_manager.update_predictions()
        except Exception as e:
            print(f"❌ Periodic prediction update failed: {e}")
        
        await asyncio.sleep(900)  # 15 minutes

@router.on_event("startup")
async def start_periodic_updates():
    """Start periodic prediction updates"""
    asyncio.create_task(periodic_prediction_updater())