import React, { useState, useEffect } from 'react';
import './ChartsPanel.css';

const ChartsPanel = ({ metrics }) => {
  const [timeRange, setTimeRange] = useState('5m');
  const [chartData, setChartData] = useState({});

  // Generate mock chart data based on current metrics
  useEffect(() => {
    if (metrics) {
      const baseTime = new Date();
      const labels = generateTimeLabels(timeRange);
      
      setChartData({
        cpu: generateLineData(metrics.cpu_percent, 15, labels),
        memory: generateLineData(metrics.memory_percent, 20, labels),
        disk: generateLineData(metrics.disk_usage || 45, 10, labels)
      });
    }
  }, [metrics, timeRange]);

  const generateTimeLabels = (range) => {
    const labels = [];
    const now = new Date();
    
    switch (range) {
      case '5m':
        for (let i = 4; i >= 0; i--) {
          const time = new Date(now - i * 60000);
          labels.push(time.getMinutes() + ':' + time.getSeconds().toString().padStart(2, '0'));
        }
        break;
      case '15m':
        for (let i = 14; i >= 0; i--) {
          const time = new Date(now - i * 60000);
          labels.push(time.getMinutes() + ':' + time.getSeconds().toString().padStart(2, '0'));
        }
        break;
      case '1h':
        for (let i = 11; i >= 0; i--) {
          const time = new Date(now - i * 300000);
          labels.push(time.getHours() + ':' + time.getMinutes().toString().padStart(2, '0'));
        }
        break;
      default:
        break;
    }
    
    return labels;
  };

  const generateLineData = (currentValue, volatility, labels) => {
    const values = [];
    let lastValue = currentValue;
    
    labels.forEach(() => {
      const change = (Math.random() - 0.5) * volatility;
      lastValue = Math.max(0, Math.min(100, lastValue + change));
      values.push(parseFloat(lastValue.toFixed(1)));
    });
    
    return {
      labels,
      values,
      current: currentValue,
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      max: Math.max(...values),
      min: Math.min(...values)
    };
  };

  const getStatus = (value, metricType) => {
    const thresholds = {
      cpu: { critical: 85, warning: 70 },
      memory: { critical: 90, warning: 75 },
      disk: { critical: 95, warning: 80 }
    };
    
    const threshold = thresholds[metricType];
    if (value >= threshold.critical) return 'critical';
    if (value >= threshold.warning) return 'warning';
    return 'normal';
  };

  const getChartColor = (status) => {
    const colors = {
      normal: '#2ecc71',
      warning: '#f39c12', 
      critical: '#e74c3c'
    };
    return colors[status];
  };

  const renderLineChart = (data, metricType) => {
    if (!data || !data.values) return null;
    
    const status = getStatus(data.current, metricType);
    const color = getChartColor(status);
    const maxValue = Math.max(...data.values, 80);
    const minValue = Math.min(...data.values, 0);
    
    // Calculate points for SVG path
    const points = data.values.map((value, index) => {
      const x = (index / (data.values.length - 1)) * 100;
      const y = 100 - ((value - minValue) / (maxValue - minValue)) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="line-chart-container">
        <svg viewBox="0 0 100 100" className="line-chart">
          {/* Grid lines */}
          <line x1="0" y1="20" x2="100" y2="20" className="grid-line" />
          <line x1="0" y1="40" x2="100" y2="40" className="grid-line" />
          <line x1="0" y1="60" x2="100" y2="60" className="grid-line" />
          <line x1="0" y1="80" x2="100" y2="80" className="grid-line" />
          
          {/* Area fill */}
          <polyline
            points={`0,100 ${points} 100,100`}
            className="chart-area"
            style={{ fill: `${color}15` }}
          />
          
          {/* Main line */}
          <polyline
            points={points}
            className="chart-line"
            style={{ stroke: color }}
          />
          
          {/* Data points */}
          {data.values.map((value, index) => {
            const x = (index / (data.values.length - 1)) * 100;
            const y = 100 - ((value - minValue) / (maxValue - minValue)) * 100;
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="1.5"
                className="data-point"
                style={{ fill: color }}
              />
            );
          })}
        </svg>
        
        {/* X-axis labels */}
        <div className="x-axis-labels">
          {data.labels.map((label, index) => (
            <span key={index} className="x-axis-label">
              {label}
            </span>
          ))}
        </div>
      </div>
    );
  };

  const getMetricIcon = (metricType) => {
    const icons = {
      cpu: '⚡',
      memory: '🧠',
      disk: '💾'
    };
    return icons[metricType];
  };

  const getMetricTitle = (metricType) => {
    const titles = {
      cpu: 'CPU Usage',
      memory: 'Memory Usage',
      disk: 'Disk Usage'
    };
    return titles[metricType];
  };

  return (
    <div className="charts-panel">
      <div className="panel-header">
        <h3 className='charts-header'>📈 Real-time System Metrics</h3>
        <div className="chart-controls">
          <div className="time-range-selector">
            <button 
              className={`time-btn ${timeRange === '5m' ? 'active' : ''}`}
              onClick={() => setTimeRange('5m')}
            >
              5m
            </button>
            <button 
              className={`time-btn ${timeRange === '15m' ? 'active' : ''}`}
              onClick={() => setTimeRange('15m')}
            >
              15m
            </button>
            <button 
              className={`time-btn ${timeRange === '1h' ? 'active' : ''}`}
              onClick={() => setTimeRange('1h')}
            >
              1h
            </button>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        {['cpu', 'memory', 'disk'].map(metricType => {
          const data = chartData[metricType];
          const status = data ? getStatus(data.current, metricType) : 'normal';
          
          return (
            <div key={metricType} className={`chart-container status-${status}`}>
              <div className="chart-header">
                <div className="chart-title">
                  <span className="metric-icon">{getMetricIcon(metricType)}</span>
                  <span>{getMetricTitle(metricType)}</span>
                </div>
                <div className="chart-stats">
                  <div className="current-value">
                    {data ? `${data.current.toFixed(1)}%` : '--%'}
                  </div>
                  <div className="stat-badges">
                    <span className="stat-badge max">↑{data ? data.max.toFixed(1) : '--'}%</span>
                    <span className="stat-badge min">↓{data ? data.min.toFixed(1) : '--'}%</span>
                  </div>
                </div>
              </div>

              <div className="chart-content">
                {data ? renderLineChart(data, metricType) : (
                  <div className="chart-placeholder">
                    <div className="loading-chart">Loading...</div>
                  </div>
                )}
              </div>

              <div className="chart-footer">
                <div className="status-indicator">
                  <div className={`status-dot ${status}`}></div>
                  <span className="status-text">
                    {status === 'critical' ? 'Critical' : 
                     status === 'warning' ? 'Warning' : 'Normal'}
                  </span>
                </div>
                <div className="trend-indicator">
                  {data && data.values.length > 1 && (
                    <>
                      <span className={`trend-arrow ${
                        data.values[data.values.length - 1] > data.values[data.values.length - 2] ? 'up' : 'down'
                      }`}>
                        {data.values[data.values.length - 1] > data.values[data.values.length - 2] ? '↗' : '↘'}
                      </span>
                      <span className="trend-text">
                        {Math.abs(data.values[data.values.length - 1] - data.values[data.values.length - 2]).toFixed(1)}%
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ChartsPanel;