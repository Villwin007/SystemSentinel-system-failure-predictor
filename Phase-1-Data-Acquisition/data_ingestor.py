import json
import time
import redis
import psycopg2
from datetime import datetime
import sys

def setup_redis():
    """Setup Redis connection with error handling"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
        r.ping()
        print("✅ Connected to Redis successfully")
        return r
    except redis.ConnectionError:
        print("❌ Could not connect to Redis. Make sure Redis is running on localhost:6379")
        sys.exit(1)

def get_db_connection():
    """Create database connection with error handling"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="monitoring",
            user="postgres",
            password="password",
            connect_timeout=10
        )
        print("✅ Connected to TimescaleDB successfully")
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to TimescaleDB: {e}")
        print("Make sure TimescaleDB is running on localhost:5432")
        sys.exit(1)

def setup_database():
    """Create the necessary table in TimescaleDB if it doesn't exist"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if table exists
        cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'system_metrics'
        );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE system_metrics (
                timestamp TIMESTAMPTZ NOT NULL,
                cpu_percent DOUBLE PRECISION,
                memory_percent DOUBLE PRECISION,
                memory_used_gb DOUBLE PRECISION,
                memory_available_gb DOUBLE PRECISION,
                disk_usage_percent DOUBLE PRECISION,
                disk_used_gb DOUBLE PRECISION,
                disk_free_gb DOUBLE PRECISION
            );
            """
            cur.execute(create_table_query)
            print("✅ Created system_metrics table")
            
            # Try to create hypertable (might fail if extension not installed, that's ok)
            try:
                cur.execute("SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);")
                print("✅ Created hypertable for system_metrics")
            except Exception as e:
                print(f"⚠️  Could not create hypertable: {e}")
                print("Continuing with regular table...")
        else:
            print("✅ system_metrics table already exists")
        
        conn.commit()
        print("✅ Database setup completed")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def check_existing_data():
    """Check if there's existing data in the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT COUNT(*) as total_records FROM system_metrics;")
        count = cur.fetchone()[0]
        print(f"📊 Existing records in database: {count}")
        
        if count > 0:
            cur.execute("SELECT MAX(timestamp) as latest FROM system_metrics;")
            latest = cur.fetchone()[0]
            print(f"📅 Latest record: {latest}")
        
        return count
    except Exception as e:
        print(f"⚠️  Error checking existing data: {e}")
        return 0
    finally:
        cur.close()
        conn.close()

def ingest_metrics():
    """Listen to Redis and ingest metrics into TimescaleDB"""
    r = setup_redis()
    
    pubsub = r.pubsub()
    pubsub.subscribe('metrics')
    
    print("🚀 Starting metrics ingestion...")
    print("Listening for metrics on Redis channel 'metrics'")
    print("Press Ctrl+C to stop")
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    
                    # Insert into database
                    conn = get_db_connection()
                    cur = conn.cursor()
                    
                    insert_query = """
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_gb, memory_available_gb, 
                     disk_usage_percent, disk_used_gb, disk_free_gb)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cur.execute(insert_query, (
                        data['timestamp'],
                        data['cpu_percent'],
                        data['memory_percent'],
                        data['memory_used_gb'],
                        data['memory_available_gb'],
                        data['disk_usage_percent'],
                        data['disk_used_gb'],
                        data['disk_free_gb']
                    ))
                    conn.commit()
                    cur.close()
                    conn.close()
                    
                    print(f"💾 Ingested: {data['timestamp'][11:19]} - CPU: {data['cpu_percent']}%")
                    
                except Exception as e:
                    print(f"⚠️  Error ingesting metrics: {e}")
                    
    except KeyboardInterrupt:
        print("\n🛑 Ingestion stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == '__main__':
    setup_database()
    existing_records = check_existing_data()
    
    if existing_records > 0:
        print(f"🎯 Continuing with existing {existing_records} records")
    
    ingest_metrics()