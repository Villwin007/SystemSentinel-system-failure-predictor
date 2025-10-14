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

def fetch_training_data():
    """Fetch recent system metrics for training"""
    conn = get_db_connection()
    
    query = """
    SELECT 
        timestamp,
        cpu_percent,
        memory_percent, 
        memory_used_gb,
        disk_usage_percent,
        disk_used_gb
    FROM system_metrics 
    ORDER BY timestamp DESC 
    LIMIT 5000
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    # Sort by timestamp (oldest first)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Fetched {len(df)} records for training")
    print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def add_synthetic_anomalies(df):
    """Add synthetic anomalies to create labeled training data"""
    df_with_anomalies = df.copy()
    df_with_anomalies['is_anomaly'] = 0  # 0 = normal, 1 = anomaly
    
    num_anomalies = min(50, len(df) // 20)  # ~5% anomalies
    
    print(f"🎯 Adding {num_anomalies} synthetic anomalies...")
    
    for _ in range(num_anomalies):
        # Randomly select a point to make anomalous
        idx = random.randint(10, len(df) - 10)
        
        # Randomly choose anomaly type
        anomaly_type = random.choice(['cpu_spike', 'memory_leak', 'disk_full'])
        
        if anomaly_type == 'cpu_spike':
            df_with_anomalies.loc[idx, 'cpu_percent'] = random.uniform(90, 100)
            df_with_anomalies.loc[idx, 'is_anomaly'] = 1
            
        elif anomaly_type == 'memory_leak':
            df_with_anomalies.loc[idx, 'memory_percent'] = random.uniform(95, 100)
            df_with_anomalies.loc[idx:idx+5, 'memory_used_gb'] *= 1.5
            df_with_anomalies.loc[idx, 'is_anomaly'] = 1
            
        elif anomaly_type == 'disk_full':
            df_with_anomalies.loc[idx, 'disk_usage_percent'] = random.uniform(95, 100)
            df_with_anomalies.loc[idx, 'is_anomaly'] = 1
    
    anomaly_count = df_with_anomalies['is_anomaly'].sum()
    print(f"✅ Added {anomaly_count} synthetic anomalies")
    
    return df_with_anomalies

def save_dataset(df, filename='training_dataset.csv'):
    """Save the dataset to CSV"""
    df.to_csv(filename, index=False)
    print(f"💾 Saved dataset to {filename} with {len(df)} records")
    print(f"📈 Anomaly distribution: {df['is_anomaly'].value_counts().to_dict()}")

if __name__ == '__main__':
    print("🚀 Generating training dataset...")
    
    # Fetch real data
    df = fetch_training_data()
    
    if len(df) > 0:
        # Add synthetic anomalies for training
        df_with_anomalies = add_synthetic_anomalies(df)
        
        # Save the dataset
        save_dataset(df_with_anomalies)
        
        print("\n📋 Dataset Summary:")
        print(f"   Total records: {len(df_with_anomalies)}")
        print(f"   Normal records: {len(df_with_anomalies[df_with_anomalies['is_anomaly'] == 0])}")
        print(f"   Anomalous records: {len(df_with_anomalies[df_with_anomalies['is_anomaly'] == 1])}")
        print(f"   Anomaly ratio: {df_with_anomalies['is_anomaly'].mean():.2%}")
    else:
        print("❌ No data found. Please make sure Phase 1 is running and has collected data.")