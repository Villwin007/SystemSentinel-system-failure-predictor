import time
import json
import psutil
import redis
from datetime import datetime

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def collect_metrics():
    """Collect system metrics and push to Redis"""
    while True:
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            timestamp = datetime.utcnow().isoformat()

            # Create metric packet
            metric = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_usage_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2)
            }

            # Push to Redis
            r.publish('metrics', json.dumps(metric))
            print(f"📊 Collected metrics: CPU {cpu_percent}%, Memory {memory.percent}%")
            
            time.sleep(5)  # Collect every 5 seconds
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            time.sleep(10)

if __name__ == '__main__':
    print("Starting Data Collector...")
    collect_metrics()