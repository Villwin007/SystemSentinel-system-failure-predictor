import React from 'react'
import './MetricsGrid.css'

const MetricCard = ({ title, value, unit, icon, color, subtitle, status }) => (
  <div className={`metric-card ${status}`}>
    <div className="metric-icon">
      {icon}
    </div>
    <div className="metric-content">
      <h3>{title}</h3>
      <div className="metric-value" style={{ color: color }}>
        {value !== null && value !== undefined ? value : '--'}{unit}
      </div>
      <p>{subtitle}</p>
    </div>
  </div>
)

const MetricsGrid = ({ metrics }) => {
  if (!metrics) {
    return (
      <div className="metrics-grid">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="metric-card loading">
            <div className="metric-skeleton"></div>
          </div>
        ))}
      </div>
    )
  }

  const getHealthScore = () => {
    const cpu = metrics.cpu_percent || 0
    const memory = metrics.memory_percent || 0
    return Math.max(0, 100 - (cpu * 0.3 + memory * 0.2))
  }

  const getHealthColor = (score) => {
    if (score >= 80) return '#10b981' // Green
    if (score >= 60) return '#f59e0b' // Yellow
    return '#ef4444' // Red
  }

  const getUsageColor = (value, thresholds = { high: 80, medium: 60 }) => {
    if (value >= thresholds.high) return '#ef4444' // Red for high usage
    if (value >= thresholds.medium) return '#f59e0b' // Yellow for medium usage
    return '#10b981' // Green for low usage
  }

  const getUsageStatus = (value, thresholds = { high: 80, medium: 60 }) => {
    if (value >= thresholds.high) return 'critical'
    if (value >= thresholds.medium) return 'warning'
    return 'normal'
  }

  const healthScore = getHealthScore()
  const healthColor = getHealthColor(healthScore)
  const healthStatus = getUsageStatus(healthScore, { high: 60, medium: 80 }) // Inverted for health score

  return (
    <div className="metrics-grid">
      <MetricCard
        title="System Health"
        value={healthScore.toFixed(1)}
        unit="%"
        icon="🏥"
        color={healthColor}
        status={healthStatus}
        subtitle="Overall system health score"
      />
      
      <MetricCard
        title="CPU Usage"
        value={metrics.cpu_percent?.toFixed(1)}
        unit="%"
        icon="⚡"
        color={getUsageColor(metrics.cpu_percent)}
        status={getUsageStatus(metrics.cpu_percent)}
        subtitle="Processor utilization"
      />
      
      <MetricCard
        title="Memory Usage"
        value={metrics.memory_percent?.toFixed(1)}
        unit="%"
        icon="💾"
        color={getUsageColor(metrics.memory_percent)}
        status={getUsageStatus(metrics.memory_percent)}
        subtitle="RAM utilization"
      />
      
      <MetricCard
        title="Disk Usage"
        value={metrics.disk_usage_percent?.toFixed(1)}
        unit="%"
        icon="💽"
        color={getUsageColor(metrics.disk_usage_percent, { high: 90, medium: 80 })}
        status={getUsageStatus(metrics.disk_usage_percent, { high: 90, medium: 80 })}
        subtitle="Storage utilization"
      />
    </div>
  )
}

export default MetricsGrid