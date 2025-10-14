from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import asyncio
import redis
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import psutil
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="AI System Monitoring API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
connected_clients = []
alerts_history = []
system_metrics_history = []

class AlertManager:
    def __init__(self):
        self.alerts = []
    
    def add_alert(self, alert_data):
        alert = {
            "id": len(self.alerts) + 1,
            "timestamp": datetime.now().isoformat(),
            "level": alert_data.get("level", "warning"),
            "title": alert_data.get("title", ""),
            "message": alert_data.get("message", ""),
            "system_metrics": alert_data.get("system_metrics", {}),
            "confidence": alert_data.get("confidence", 0.0),
            "resolved": False
        }
        self.alerts.append(alert)
        
        # Broadcast to connected clients
        asyncio.create_task(broadcast_alert(alert))
        
        return alert

alert_manager = AlertManager()

class SystemMetrics(BaseModel):
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    is_anomaly: bool = False
    anomaly_confidence: float = 0.0

class AlertRequest(BaseModel):
    level: str  # "info", "warning", "critical"
    title: str
    message: str
    system_metrics: Optional[Dict] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

async def broadcast_alert(alert):
    """Broadcast new alert to all connected clients"""
    message = json.dumps({
        "type": "new_alert",
        "data": alert
    })
    await manager.broadcast(message)

async def broadcast_metrics(metrics):
    """Broadcast system metrics to all connected clients"""
    message = json.dumps({
        "type": "system_metrics",
        "data": metrics
    })
    await manager.broadcast(message)

@app.get("/")
async def root():
    return {"message": "AI System Monitoring API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/system/current")
async def get_current_system_metrics():
    """Get current system metrics"""
    try:
        # Get real system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
        
        return metrics
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/alerts")
async def get_alerts(limit: int = 50, resolved: bool = False):
    """Get alerts history"""
    alerts = alert_manager.alerts
    if not resolved:
        alerts = [alert for alert in alerts if not alert["resolved"]]
    
    return alerts[-limit:]

@app.post("/api/alerts")
async def create_alert(alert: AlertRequest):
    """Create a new alert"""
    new_alert = alert_manager.add_alert({
        "level": alert.level,
        "title": alert.title,
        "message": alert.message,
        "system_metrics": alert.system_metrics
    })
    
    return {"status": "alert_created", "alert": new_alert}

@app.put("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int):
    """Mark an alert as resolved"""
    for alert in alert_manager.alerts:
        if alert["id"] == alert_id:
            alert["resolved"] = True
            alert["resolved_at"] = datetime.now().isoformat()
            
            # Broadcast resolution
            asyncio.create_task(broadcast_alert({
                "type": "alert_resolved",
                "alert_id": alert_id
            }))
            
            return {"status": "alert_resolved", "alert_id": alert_id}
    
    return {"error": "Alert not found"}

@app.get("/api/metrics/history")
async def get_metrics_history(hours: int = 24):
    """Get historical system metrics"""
    # In production, this would query your TimescaleDB
    # For now, return mock data or recent history
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    # Generate mock historical data
    history = []
    current_time = start_time
    
    while current_time <= end_time:
        history.append({
            "timestamp": current_time.isoformat(),
            "cpu_percent": np.random.uniform(20, 80),
            "memory_percent": np.random.uniform(50, 90),
            "disk_usage_percent": np.random.uniform(30, 70),
            "is_anomaly": np.random.random() > 0.95
        })
        current_time += timedelta(minutes=5)
    
    return history

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    total_alerts = len(alert_manager.alerts)
    critical_alerts = len([a for a in alert_manager.alerts if a["level"] == "critical"])
    active_alerts = len([a for a in alert_manager.alerts if not a["resolved"]])
    
    # Calculate system health score (mock for now)
    health_score = max(0, 100 - (active_alerts * 10))
    
    return {
        "health_score": health_score,
        "total_alerts": total_alerts,
        "critical_alerts": critical_alerts,
        "active_alerts": active_alerts,
        "system_uptime": "99.8%",  # Mock data
        "avg_response_time": "45ms"  # Mock data
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(10)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def simulate_live_metrics():
    """Simulate live metrics broadcasting (replace with real data from Redis)"""
    while True:
        try:
            metrics = await get_current_system_metrics()
            if "error" not in metrics:
                await broadcast_metrics(metrics)
                system_metrics_history.append(metrics)
                
                # Keep only last 1000 points
                if len(system_metrics_history) > 1000:
                    system_metrics_history.pop(0)
        
        except Exception as e:
            print(f"Error in metrics simulation: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(simulate_live_metrics())
    
    # Create a sample critical alert to demonstrate
    alert_manager.add_alert({
        "level": "critical",
        "title": "System Monitoring Started",
        "message": "AI monitoring system is now active and monitoring system metrics",
        "system_metrics": await get_current_system_metrics()
    })

# Frontend HTML (simple dashboard)
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI System Monitoring Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .dashboard { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .alerts-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .alert { padding: 10px; margin: 10px 0; border-left: 4px solid #ff6b6b; background: #fff5f5; }
            .alert.critical { border-color: #ff6b6b; background: #fff5f5; }
            .alert.warning { border-color: #ffd93d; background: #fff9e6; }
            .alert.info { border-color: #6bcff6; background: #f0f9ff; }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>🚀 AI System Monitoring Dashboard</h1>
                <p>Real-time system health monitoring with AI-powered anomaly detection</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>System Health</h3>
                    <div id="health-score" style="font-size: 2em; color: #4ecdc4;">95%</div>
                </div>
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <div id="cpu-usage" style="font-size: 2em; color: #45b7d1;">--%</div>
                </div>
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <div id="memory-usage" style="font-size: 2em; color: #96ceb4;">--%</div>
                </div>
                <div class="metric-card">
                    <h3>Active Alerts</h3>
                    <div id="active-alerts" style="font-size: 2em; color: #ff6b6b;">--</div>
                </div>
            </div>
            
            <div class="alerts-panel">
                <h3>Recent Alerts</h3>
                <div id="alerts-container"></div>
            </div>
        </div>

        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'system_metrics') {
                    document.getElementById('cpu-usage').textContent = data.data.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = data.data.memory_percent.toFixed(1) + '%';
                }
                else if (data.type === 'new_alert') {
                    addAlertToUI(data.data);
                }
            };

            function addAlertToUI(alert) {
                const container = document.getElementById('alerts-container');
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert ' + alert.level;
                alertDiv.innerHTML = `
                    <strong>${alert.title}</strong>
                    <p>${alert.message}</p>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                `;
                container.insertBefore(alertDiv, container.firstChild);
                
                // Update active alerts count
                document.getElementById('active-alerts').textContent = 
                    parseInt(document.getElementById('active-alerts').textContent || '0') + 1;
            }

            // Load initial data
            fetch('/api/analytics/summary')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('health-score').textContent = data.health_score + '%';
                    document.getElementById('active-alerts').textContent = data.active_alerts;
                });

            fetch('/api/alerts?limit=10')
                .then(r => r.json())
                .then(alerts => {
                    alerts.forEach(alert => addAlertToUI(alert));
                });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)