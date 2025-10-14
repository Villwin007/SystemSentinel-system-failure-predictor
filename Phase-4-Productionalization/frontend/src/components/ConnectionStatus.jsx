import React from 'react';
import './ConnectionStatus.css';

const ConnectionStatus = ({ isConnected }) => {
  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <div className="status-indicator"></div>
      <span className="status-text">
        {isConnected ? 'Connected to AI Monitor' : 'Disconnected'}
      </span>
      {!isConnected && (
        <span className="reconnecting-text">Reconnecting...</span>
      )}
    </div>
  );
};

export default ConnectionStatus;