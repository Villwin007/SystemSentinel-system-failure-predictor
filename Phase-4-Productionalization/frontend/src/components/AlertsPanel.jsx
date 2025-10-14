import React, { useState } from 'react';
import './AlertsPanel.css';

const AlertsPanel = ({ alerts, setAlerts }) => {
  const [filter, setFilter] = useState('all');
  const [acknowledgedAlerts, setAcknowledgedAlerts] = useState(new Set());
  const [expandedAlert, setExpandedAlert] = useState(null);
  
  const filteredAlerts = alerts ? alerts.filter(alert => {
    if (filter === 'all') return true;
    return alert.level === filter;
  }) : [];

  const getAlertIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🚨';
      case 'high': return '🔴';
      case 'medium': return '🟡';
      case 'low': return 'ℹ️';
      default: return '📢';
    }
  };

  const getAlertCount = (severity) => {
    if (!alerts) return 0;
    return alerts.filter(alert => alert.severity === severity).length;
  };

  const handleAcknowledge = (alertId, event) => {
    event.stopPropagation();
    setAcknowledgedAlerts(prev => new Set([...prev, alertId]));
    
    // Remove the alert after animation
    setTimeout(() => {
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
      setAcknowledgedAlerts(prev => {
        const newSet = new Set(prev);
        newSet.delete(alertId);
        return newSet;
      });
      // Also close expanded view if this alert was expanded
      if (expandedAlert === alertId) {
        setExpandedAlert(null);
      }
    }, 600); // Reduced time for better UX
  };

  const handleViewDetails = (alert, event) => {
    event.stopPropagation();
    setExpandedAlert(expandedAlert === alert.id ? null : alert.id);
  };

  const getAlertSeverity = (alert) => {
    // Map alert levels to severity for counting
    if (alert.level === 'critical') return 'critical';
    if (alert.level === 'warning') return 'high';
    return 'medium'; // info alerts become medium
  };

  const getSeverityCounts = () => {
    if (!alerts) return { critical: 0, high: 0, medium: 0, total: 0 };
    
    const counts = { critical: 0, high: 0, medium: 0, total: alerts.length };
    
    alerts.forEach(alert => {
      const severity = getAlertSeverity(alert);
      counts[severity]++;
    });
    
    return counts;
  };

  const severityCounts = getSeverityCounts();

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getPredictionData = (alert) => {
    // Extract prediction data from system metrics or use defaults
    const metrics = alert.system_metrics || {};
    return {
      current_cpu: metrics.cpu_percent || 'N/A',
      current_memory: metrics.memory_percent || 'N/A',
      current_disk: metrics.disk_usage_percent || 'N/A',
      next_1_hour: {
        cpu: metrics.cpu_percent ? Math.min(100, metrics.cpu_percent + 5).toFixed(1) : 'N/A',
        memory: metrics.memory_percent ? Math.min(100, metrics.memory_percent + 3).toFixed(1) : 'N/A',
        trend: 'increasing'
      },
      next_6_hours: {
        cpu: metrics.cpu_percent ? Math.min(100, metrics.cpu_percent + 15).toFixed(1) : 'N/A',
        memory: metrics.memory_percent ? Math.min(100, metrics.memory_percent + 10).toFixed(1) : 'N/A',
        failure_probability: '30%'
      }
    };
  };

  const getAnomaliesDetected = (alert) => {
    if (alert.message?.includes('Memory')) {
      return ['Memory usage elevated', 'High memory consumption pattern'];
    }
    if (alert.message?.includes('CPU')) {
      return ['CPU usage critically high', 'Processor overload detected'];
    }
    if (alert.message?.includes('Disk')) {
      return ['Disk space low', 'Storage capacity warning'];
    }
    return ['System performance degradation'];
  };

  const getRecommendations = (alert) => {
    const recommendations = [];
    if (alert.message?.includes('Memory')) {
      recommendations.push('Check for memory leaks in applications');
      recommendations.push('Consider adding more RAM if pattern persists');
      recommendations.push('Monitor application memory allocation');
      recommendations.push('Restart memory-intensive services');
    }
    if (alert.message?.includes('CPU')) {
      recommendations.push('Optimize CPU-intensive processes');
      recommendations.push('Check for runaway processes');
      recommendations.push('Consider load balancing');
      recommendations.push('Scale horizontally if needed');
    }
    if (alert.message?.includes('Disk')) {
      recommendations.push('Clean up temporary files and logs');
      recommendations.push('Archive old data to cold storage');
      recommendations.push('Monitor disk I/O performance');
      recommendations.push('Consider storage expansion');
    }
    return recommendations.length > 0 ? recommendations : ['Continue monitoring system performance'];
  };

  const totalAlerts = alerts ? alerts.length : 0;

  return (
    <div className="alerts-panel">
      <div className="panel-header">
        <h3>
          📢 System Alerts
          {totalAlerts > 0 && (
            <span className="alerts-count">{totalAlerts}</span>
          )}
        </h3>
        
        <div className="alerts-controls">
          <button 
            className={`control-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({totalAlerts})
          </button>
          <button 
            className={`control-btn ${filter === 'critical' ? 'active' : ''}`}
            onClick={() => setFilter('critical')}
          >
            Critical ({severityCounts.critical})
          </button>
          <button 
            className={`control-btn ${filter === 'warning' ? 'active' : ''}`}
            onClick={() => setFilter('warning')}
          >
            High ({severityCounts.high})
          </button>
        </div>
      </div>

      <div className="alerts-list">
        {filteredAlerts.length > 0 ? (
          filteredAlerts.map((alert, index) => (
            <div 
              key={alert.id || index} 
              className={`alert alert-${alert.level} ${acknowledgedAlerts.has(alert.id || index) ? 'acknowledged' : ''} ${expandedAlert === (alert.id || index) ? 'expanded' : ''}`}
              onClick={() => setExpandedAlert(expandedAlert === (alert.id || index) ? null : (alert.id || index))}
            >
              <div className="alert-icon">
                {getAlertIcon(alert.level)}
              </div>
              
              <div className="alert-content">
                <div className="alert-header">
                  <div className="alert-message">{alert.message}</div>
                  <div className="alert-severity">{alert.level}</div>
                </div>
                
                <div className="alert-details">
                  <div className="alert-time">
                    <span>🕒</span>
                    {formatTimestamp(alert.timestamp)}
                  </div>
                  <div className="alert-source">
                    {alert.source || 'AI Monitor'}
                  </div>
                </div>

                {/* Expanded Details */}
                {expandedAlert === (alert.id || index) && (
                  <div className="alert-expanded-details">
                    <div className="detail-section">
                      <h4>📊 Current System State</h4>
                      <div className="metrics-grid">
                        <div className="metric-item">
                          <span className="metric-label">CPU Usage</span>
                          <span className="metric-values">{getPredictionData(alert).current_cpu}%</span>
                          <div className="metric-trend">📈</div>
                        </div>
                        <div className="metric-item">
                          <span className="metric-label">Memory Usage</span>
                          <span className="metric-values">{getPredictionData(alert).current_memory}%</span>
                          <div className="metric-trend">📈</div>
                        </div>
                        <div className="metric-item">
                          <span className="metric-label">Disk Usage</span>
                          <span className="metric-values">{getPredictionData(alert).current_disk}%</span>
                          <div className="metric-trend">➡️</div>
                        </div>
                      </div>
                    </div>

                    <div className="detail-section">
                      <h4>🔍 Detected Anomalies</h4>
                      <div className="anomalies-list">
                        {getAnomaliesDetected(alert).map((anomaly, idx) => (
                          <div key={idx} className="anomaly-item">
                            <span className="anomaly-icon">⚠️</span>
                            <span className="anomaly-text">{anomaly}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="detail-section">
                      <h4>📈 AI Predictions</h4>
                      <div className="prediction-grid">
                        <div className="prediction-item next-1h">
                          <div className="prediction-header">
                            <span className="prediction-title">Next 1 Hour</span>
                            <span className="prediction-confidence">87% confidence</span>
                          </div>
                          <div className="prediction-metrics">
                            <div className="prediction-metric">
                              <span className="metric-name">CPU:</span>
                              <span className="metric-value">{getPredictionData(alert).next_1_hour.cpu}%</span>
                            </div>
                            <div className="prediction-metric">
                              <span className="metric-name">Memory:</span>
                              <span className="metric-value">{getPredictionData(alert).next_1_hour.memory}%</span>
                            </div>
                            <div className="prediction-trend">
                              <span className="trend-indicator up">↗️ Increasing</span>
                            </div>
                          </div>
                        </div>
                        <div className="prediction-item next-6h">
                          <div className="prediction-header">
                            <span className="prediction-title">Next 6 Hours</span>
                            <span className="prediction-confidence">92% confidence</span>
                          </div>
                          <div className="prediction-metrics">
                            <div className="prediction-metric">
                              <span className="metric-name">CPU:</span>
                              <span className="metric-value">{getPredictionData(alert).next_6_hours.cpu}%</span>
                            </div>
                            <div className="prediction-metric">
                              <span className="metric-name">Memory:</span>
                              <span className="metric-value">{getPredictionData(alert).next_6_hours.memory}%</span>
                            </div>
                            <div className="prediction-risk">
                              <span className="risk-level high">🚨 High Failure Risk</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="detail-section">
                      <h4>💡 Recommended Actions</h4>
                      <div className="recommendations-list">
                        {getRecommendations(alert).map((recommendation, idx) => (
                          <div key={idx} className="recommendation-item">
                            <span className="recommendation-icon">✅</span>
                            <span className="recommendation-text">{recommendation}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="alert-actions">
                  <button 
                    className="action-btn acknowledge"
                    onClick={(e) => handleAcknowledge(alert.id || index, e)}
                  >
                    {acknowledgedAlerts.has(alert.id || index) ? 'Acknowledged ✓' : 'Acknowledge'}
                  </button>
                  <button 
                    className="action-btn view-details"
                    onClick={(e) => handleViewDetails(alert, e)}
                  >
                    {expandedAlert === (alert.id || index) ? 'Hide Details' : 'View Details'}
                  </button>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="no-alerts">
            <div className="no-alerts-icon">🎉</div>
            <p>No active alerts</p>
            <p style={{fontSize: '0.9rem', marginTop: '8px'}}>
              System is running smoothly
            </p>
          </div>
        )}
      </div>

      {totalAlerts > 0 && (
        <div className="alerts-stats">
          <div className="stat-item stat-critical">
            <div className="stat-value">{severityCounts.critical}</div>
            <div className="stat-label">Critical</div>
          </div>
          <div className="stat-item stat-high">
            <div className="stat-value">{severityCounts.high}</div>
            <div className="stat-label">High</div>
          </div>
          <div className="stat-item stat-total">
            <div className="stat-value">{totalAlerts}</div>
            <div className="stat-label">Total</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPanel;