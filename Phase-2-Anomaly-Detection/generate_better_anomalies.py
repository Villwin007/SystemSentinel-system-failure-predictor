import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import random

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="monitoring", 
        user="postgres",
        password="password"
    )

def generate_realistic_anomalies():
    """Generate more realistic and subtle anomalies"""
    print("🎯 Generating Realistic Anomalies...")
    
    # Fetch real data
    conn = get_db_connection()
    query = "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 5000"
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Base dataset: {len(df)} records")
    
    # Add more subtle and realistic anomalies
    df_with_anomalies = df.copy()
    df_with_anomalies['is_anomaly'] = 0
    
    num_anomalies = min(200, len(df) // 25)  # More anomalies, but still rare
    
    print(f"🎯 Adding {num_anomalies} realistic anomalies...")
    
    anomaly_count = 0
    
    for _ in range(num_anomalies * 2):  # Try more times to get enough
        if anomaly_count >= num_anomalies:
            break
            
        idx = random.randint(20, len(df) - 20)
        
        # Make anomalies more subtle and realistic
        anomaly_type = random.choice([
            'gradual_cpu_increase', 'memory_leak', 'disk_fill', 
            'network_spike', 'mixed_issue', 'temporary_spike'
        ])
        
        try:
            if anomaly_type == 'gradual_cpu_increase':
                # Gradual CPU increase over 5 points
                for i in range(5):
                    if idx + i < len(df):
                        increase = random.uniform(1.1, 1.5)  # 10-50% increase
                        df_with_anomalies.loc[idx + i, 'cpu_percent'] = min(
                            df_with_anomalies.loc[idx + i, 'cpu_percent'] * increase, 95
                        )
                        df_with_anomalies.loc[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
                
            elif anomaly_type == 'memory_leak':
                # Memory gradually increasing
                for i in range(8):
                    if idx + i < len(df):
                        increase = random.uniform(1.05, 1.2)  # 5-20% increase
                        df_with_anomalies.loc[idx + i, 'memory_percent'] = min(
                            df_with_anomalies.loc[idx + i, 'memory_percent'] * increase, 98
                        )
                        df_with_anomalies.loc[idx + i, 'memory_used_gb'] *= increase
                        df_with_anomalies.loc[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
                
            elif anomaly_type == 'disk_fill':
                # Disk gradually filling up
                for i in range(10):
                    if idx + i < len(df):
                        increase = random.uniform(1.02, 1.1)  # 2-10% increase
                        df_with_anomalies.loc[idx + i, 'disk_usage_percent'] = min(
                            df_with_anomalies.loc[idx + i, 'disk_usage_percent'] * increase, 99
                        )
                        df_with_anomalies.loc[idx + i, 'disk_used_gb'] *= increase
                        df_with_anomalies.loc[idx + i, 'is_anomaly'] = 1
                anomaly_count += 1
                
            elif anomaly_type == 'temporary_spike':
                # Short temporary spike
                spike_factor = random.uniform(1.5, 3.0)
                df_with_anomalies.loc[idx, 'cpu_percent'] = min(
                    df_with_anomalies.loc[idx, 'cpu_percent'] * spike_factor, 100
                )
                df_with_anomalies.loc[idx, 'is_anomaly'] = 1
                anomaly_count += 1
                
            elif anomaly_type == 'mixed_issue':
                # Multiple metrics slightly off
                df_with_anomalies.loc[idx, 'cpu_percent'] = min(
                    df_with_anomalies.loc[idx, 'cpu_percent'] * 1.3, 85
                )
                df_with_anomalies.loc[idx, 'memory_percent'] = min(
                    df_with_anomalies.loc[idx, 'memory_percent'] * 1.2, 80
                )
                df_with_anomalies.loc[idx, 'is_anomaly'] = 1
                anomaly_count += 1
                
        except Exception as e:
            continue
    
    final_anomaly_count = df_with_anomalies['is_anomaly'].sum()
    print(f"✅ Added {final_anomaly_count} realistic anomalies")
    print(f"📈 Anomaly rate: {final_anomaly_count/len(df_with_anomalies):.2%}")
    
    # Save new dataset
    df_with_anomalies.to_csv('training_dataset_realistic.csv', index=False)
    print("💾 Saved realistic dataset: training_dataset_realistic.csv")
    
    return df_with_anomalies

if __name__ == '__main__':
    generate_realistic_anomalies()