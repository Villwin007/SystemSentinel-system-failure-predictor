// import React, { useState, useEffect } from 'react'
// import MetricsGrid from './MetricsGrid'
// import AlertsPanel from './AlertsPanel'
// import ForecastingPanel from './ForecastingPanel'
// import ChartsPanel from './ChartsPanel'
// import useWebSocket from '../hooks/useWebSocket'
// import './Dashboard.css'

// const Dashboard = ({ onConnectionChange }) => {
//   const [systemMetrics, setSystemMetrics] = useState(null)
//   const [alerts, setAlerts] = useState([])
//   const [forecastingData, setForecastingData] = useState(null)

//   const { lastMessage, connectionStatus } = useWebSocket('ws://localhost:8000/ws')

//   // Update parent connection status
//   useEffect(() => {
//     onConnectionChange(connectionStatus)
//   }, [connectionStatus, onConnectionChange])

//   // Handle incoming WebSocket messages
//   useEffect(() => {
//     if (lastMessage) {
//       const data = JSON.parse(lastMessage.data)
      
//       switch (data.type) {
//         case 'system_metrics':
//           setSystemMetrics(data.data)
//           // Update forecasting when new metrics arrive
//           loadForecastingData()
//           break
//         case 'new_alert':
//           setAlerts(prev => [data.data, ...prev.slice(0, 19)]) // Keep last 20 alerts
//           break
//         default:
//           break
//       }
//     }
//   }, [lastMessage])

//   // Load initial data and forecasting
//   useEffect(() => {
//     loadInitialData()
//     loadForecastingData()
    
//     // Refresh forecasting every 2 minutes to get updated predictions
//     const interval = setInterval(loadForecastingData, 120000)
//     return () => clearInterval(interval)
//   }, [])

//   const loadInitialData = async () => {
//     try {
//       console.log('📥 Loading initial dashboard data...')
//       const [alertsRes, metricsRes] = await Promise.all([
//         fetch('/api/alerts?limit=20'),
//         fetch('/api/system/current')
//       ])
      
//       if (!alertsRes.ok) throw new Error('Alerts API failed')
//       if (!metricsRes.ok) throw new Error('Metrics API failed')
      
//       const alertsData = await alertsRes.json()
//       const metricsData = await metricsRes.json()
      
//       setAlerts(alertsData)
//       if (!systemMetrics) setSystemMetrics(metricsData)
      
//       console.log('✅ Initial data loaded successfully')
//     } catch (error) {
//       console.error('❌ Error loading initial data:', error)
//     }
//   }

//   const loadForecastingData = async () => {
//     try {
//       console.log('🔮 Loading real AI forecasting predictions...');

//       // Use absolute URL to avoid CORS issues
//       const baseUrl = 'http://localhost:8000';
//       const response = await fetch(`${baseUrl}/api/forecasting/predictions`);

//       console.log('📡 Response status:', response.status, response.statusText);

//       if (!response.ok) {
//         const errorText = await response.text();
//         console.error('❌ API Error:', response.status, errorText);
//         throw new Error(`HTTP ${response.status}: ${errorText}`);
//       }

//       const data = await response.json();
//       console.log('✅ REAL forecasting data received:', data);
      
//       // Success! Use the real data
//       setForecastingData({
//         ...data,
//         fallback_mode: false // Explicitly mark as real data
//       });
      
//     } catch (error) {
//       console.error('❌ Error loading forecasting data:', error);
      
//       // Try a direct test to see if it's a CORS issue
//       try {
//         console.log('🔄 Testing direct API access...');
//         const testResponse = await fetch('http://localhost:8000/api/debug/forecasting');
//         if (testResponse.ok) {
//           const testData = await testResponse.json();
//           console.log('✅ Debug endpoint works:', testData);
//         }
//       } catch (testError) {
//         console.error('❌ Debug endpoint also failed:', testError);
//       }

//       // Create intelligent fallback based on current system state
//       const currentMetrics = systemMetrics || await fetchCurrentMetrics();
//       const fallbackData = createIntelligentFallback(currentMetrics);
//       setForecastingData(fallbackData);
//     }
//   };

//   // Helper function to fetch current metrics if not available
//   const fetchCurrentMetrics = async () => {
//     try {
//       const response = await fetch('http://localhost:8000/api/system/current');
//       if (response.ok) {
//         return await response.json();
//       }
//     } catch (error) {
//       console.error('Error fetching current metrics:', error);
//     }
//     return null;
//   };

//   // Improved intelligent fallback
//   const createIntelligentFallback = (metrics) => {
//     if (!metrics) {
//       return {
//         fallback_mode: true,
//         overall_risk: 'unknown',
//         message: 'Unable to connect to forecasting service',
//         timestamp: new Date().toISOString()
//       };
//     }

//     const { cpu_percent, memory_percent, disk_usage_percent } = metrics;

//     // Calculate real risk based on actual metrics
//     let overallRisk = 'low';
//     if (memory_percent > 90 || cpu_percent > 90) overallRisk = 'high';
//     else if (memory_percent > 85 || cpu_percent > 80) overallRisk = 'medium';

//     return {
//       fallback_mode: true,
//       overall_risk: overallRisk,
//       confidence: 0.7,
//       message: `Using system metrics: CPU ${cpu_percent}%, Memory ${memory_percent}%`,
//       component_risks: {
//         cpu_performance: cpu_percent > 85 ? 'high' : cpu_percent > 70 ? 'medium' : 'low',
//         memory_usage: memory_percent > 90 ? 'high' : memory_percent > 75 ? 'medium' : 'low',
//         disk_io: disk_usage_percent > 95 ? 'high' : disk_usage_percent > 80 ? 'medium' : 'low'
//       },
//       timestamp: new Date().toISOString(),
//       system_metrics: metrics
//     };
//   };

//   // Test all API endpoints on component mount
//   useEffect(() => {
//     const testEndpoints = async () => {
//       if (process.env.NODE_ENV === 'development') {
//         console.log('🧪 Testing API endpoints...')
//         const endpoints = [
//           '/api/forecasting/predictions',
//           '/api/forecasting/resource-forecast',
//           '/api/forecasting/failure-risk',
//           '/api/forecasting/status',
//           '/api/system/current',
//           '/api/alerts?limit=5'
//         ]
        
//         for (const endpoint of endpoints) {
//           try {
//             const response = await fetch(endpoint)
//             const data = await response.json()
//             console.log(`✅ ${endpoint}:`, data)
//           } catch (error) {
//             console.log(`❌ ${endpoint}:`, error.message)
//           }
//         }
//       }
//     }
    
//     testEndpoints()
//   }, [])

//   // Refresh forecasting when system metrics change significantly
//   useEffect(() => {
//     if (systemMetrics) {
//       const shouldRefresh = 
//         systemMetrics.cpu_percent > 80 || 
//         systemMetrics.memory_percent > 85 ||
//         systemMetrics.disk_usage_percent > 90
      
//       if (shouldRefresh) {
//         console.log('🔄 Significant metric change detected, refreshing forecasts...')
//         loadForecastingData()
//       }
//     }
//   }, [systemMetrics])

//   return (
//     <div className="dashboard">
//       <header className="dashboard-header">
//         <div className="header-content">
//           <h1>🧠 AI System Monitoring Dashboard</h1>
//           <p>Real-time system health monitoring with AI-powered anomaly detection and forecasting</p>
//           <div className="sub-header">
//             <span className="model-info">BiLSTM Model • 97.4% Accuracy • 6-Hour Predictions</span>
//           </div>
//         </div>
//         <div className="header-stats">
//           {systemMetrics && (
//             <div className="live-indicator">
//               <div className="pulse"></div>
//               <span>LIVE</span>
//               <span className="update-time">
//                 Updated: {new Date().toLocaleTimeString()}
//               </span>
//             </div>
//           )}
//         </div>
//       </header>

//       <MetricsGrid metrics={systemMetrics} />

//       <div className="dashboard-content">
//         <div className="content-main">
//           <ChartsPanel metrics={systemMetrics} />
//           <AlertsPanel alerts={alerts} setAlerts={setAlerts}/>
//         </div>
        
//         <div className="content-sidebar">
//           <ForecastingPanel 
//             data={forecastingData} 
//             systemMetrics={systemMetrics}
//           />
//         </div>
//       </div>

//       {/* Debug info in development */}
//       {process.env.NODE_ENV === 'development' && forecastingData && (
//         <div className="debug-info">
//           <details>
//             <summary>Debug: Forecasting Data</summary>
//             <pre>{JSON.stringify(forecastingData, null, 2)}</pre>
//           </details>
//         </div>
//       )}
//     </div>
//   )
// }

// export default Dashboard


import React, { useState, useEffect } from 'react'
import MetricsGrid from './MetricsGrid'
import AlertsPanel from './AlertsPanel'
import ForecastingPanel from './ForecastingPanel'
import ChartsPanel from './ChartsPanel'
import useWebSocket from '../hooks/useWebSocket'
import './Dashboard.css'

const Dashboard = ({ onConnectionChange }) => {
  const [systemMetrics, setSystemMetrics] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [forecastingData, setForecastingData] = useState(null)

  // Use absolute URL for WebSocket
  const { lastMessage, connectionStatus } = useWebSocket('ws://localhost:8000/ws')

  // Update parent connection status
  useEffect(() => {
    onConnectionChange(connectionStatus)
  }, [connectionStatus, onConnectionChange])

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data)
      
      switch (data.type) {
        case 'system_metrics':
          setSystemMetrics(data.data)
          // Update forecasting when new metrics arrive
          loadForecastingData()
          break
        case 'new_alert':
          setAlerts(prev => [data.data, ...prev.slice(0, 19)]) // Keep last 20 alerts
          break
        default:
          break
      }
    }
  }, [lastMessage])

  // API base URL - use absolute URL to ensure we hit the FastAPI backend
  const API_BASE_URL = 'http://localhost:8000'

  // Load initial data and forecasting
  useEffect(() => {
    loadInitialData()
    loadForecastingData()
    
    // Refresh forecasting every 2 minutes to get updated predictions
    const interval = setInterval(loadForecastingData, 120000)
    return () => clearInterval(interval)
  }, [])

  const loadInitialData = async () => {
    try {
      console.log('📥 Loading initial dashboard data...')
      
      // Use absolute URLs
      const [alertsRes, metricsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/alerts?limit=20`),
        fetch(`${API_BASE_URL}/api/system/current`)
      ])
      
      if (!alertsRes.ok) throw new Error('Alerts API failed')
      if (!metricsRes.ok) throw new Error('Metrics API failed')
      
      const alertsData = await alertsRes.json()
      const metricsData = await metricsRes.json()
      
      setAlerts(alertsData)
      if (!systemMetrics) setSystemMetrics(metricsData)
      
      console.log('✅ Initial data loaded successfully')
    } catch (error) {
      console.error('❌ Error loading initial data:', error)
    }
  }

  const loadForecastingData = async () => {
    try {
      console.log('🔮 Loading real AI forecasting predictions...');

      // Use absolute URL
      const response = await fetch(`${API_BASE_URL}/api/forecasting/predictions`);

      console.log('📡 Response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ API Error:', response.status, errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log('✅ REAL forecasting data received:', data);
      
      // Success! Use the real data
      setForecastingData({
        ...data,
        fallback_mode: false // Explicitly mark as real data
      });
      
    } catch (error) {
      console.error('❌ Error loading forecasting data:', error);
      
      // Create intelligent fallback based on current system state
      const currentMetrics = systemMetrics || await fetchCurrentMetrics();
      const fallbackData = createIntelligentFallback(currentMetrics);
      setForecastingData(fallbackData);
    }
  };

  // Helper function to fetch current metrics if not available
  const fetchCurrentMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/system/current`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Error fetching current metrics:', error);
    }
    return null;
  };

  // Improved intelligent fallback
  const createIntelligentFallback = (metrics) => {
    if (!metrics) {
      return {
        fallback_mode: true,
        overall_risk: 'unknown',
        message: 'Unable to connect to forecasting service',
        timestamp: new Date().toISOString()
      };
    }

    const { cpu_percent, memory_percent, disk_usage_percent } = metrics;

    // Calculate real risk based on actual metrics
    let overallRisk = 'low';
    if (memory_percent > 90 || cpu_percent > 90) overallRisk = 'high';
    else if (memory_percent > 85 || cpu_percent > 80) overallRisk = 'medium';

    return {
      fallback_mode: true,
      overall_risk: overallRisk,
      confidence: 0.7,
      message: `Using system metrics: CPU ${cpu_percent}%, Memory ${memory_percent}%`,
      component_risks: {
        cpu_performance: cpu_percent > 85 ? 'high' : cpu_percent > 70 ? 'medium' : 'low',
        memory_usage: memory_percent > 90 ? 'high' : memory_percent > 75 ? 'medium' : 'low',
        disk_io: disk_usage_percent > 95 ? 'high' : disk_usage_percent > 80 ? 'medium' : 'low'
      },
      timestamp: new Date().toISOString(),
      system_metrics: metrics
    };
  };

  // Test all API endpoints on component mount
  useEffect(() => {
    const testEndpoints = async () => {
      if (process.env.NODE_ENV === 'development') {
        console.log('🧪 Testing API endpoints with absolute URLs...')
        const endpoints = [
          `${API_BASE_URL}/api/forecasting/predictions`,
          `${API_BASE_URL}/api/forecasting/resource-forecast`,
          `${API_BASE_URL}/api/forecasting/failure-risk`,
          `${API_BASE_URL}/api/forecasting/status`,
          `${API_BASE_URL}/api/system/current`,
          `${API_BASE_URL}/api/alerts?limit=5`
        ]
        
        for (const endpoint of endpoints) {
          try {
            const response = await fetch(endpoint)
            const data = await response.json()
            console.log(`✅ ${endpoint}:`, data)
          } catch (error) {
            console.log(`❌ ${endpoint}:`, error.message)
          }
        }
      }
    }
    
    testEndpoints()
  }, [])

  // Refresh forecasting when system metrics change significantly
  useEffect(() => {
    if (systemMetrics) {
      const shouldRefresh = 
        systemMetrics.cpu_percent > 80 || 
        systemMetrics.memory_percent > 85 ||
        systemMetrics.disk_usage_percent > 90
      
      if (shouldRefresh) {
        console.log('🔄 Significant metric change detected, refreshing forecasts...')
        loadForecastingData()
      }
    }
  }, [systemMetrics])

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <h1>🧠 AI System Monitoring Dashboard</h1>
          <p>Real-time system health monitoring with AI-powered anomaly detection and forecasting</p>
          <div className="sub-header">
            <span className="model-info">BiLSTM Model • 97.4% Accuracy • 6-Hour Predictions</span>
          </div>
        </div>
        <div className="header-stats">
          {systemMetrics && (
            <div className="live-indicator">
              <div className="pulse"></div>
              <span>LIVE</span>
              <span className="update-time">
                Updated: {new Date().toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
      </header>

      <MetricsGrid metrics={systemMetrics} />

      <div className="dashboard-content">
        <div className="content-main">
          <ChartsPanel metrics={systemMetrics} />
          <AlertsPanel alerts={alerts} setAlerts={setAlerts} />
        </div>
        
        <div className="content-sidebar">
          <ForecastingPanel 
            data={forecastingData} 
            systemMetrics={systemMetrics}
          />
        </div>
      </div>

      {/* Debug info in development */}
      {process.env.NODE_ENV === 'development' && forecastingData && (
        <div className="debug-info">
          <details>
            <summary>Debug: Forecasting Data</summary>
            <pre>{JSON.stringify(forecastingData, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

export default Dashboard