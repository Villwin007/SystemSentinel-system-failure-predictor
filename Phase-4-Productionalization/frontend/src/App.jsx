import React, { useState, useEffect } from "react";
import Dashboard from "./components/Dashboard";
import ConnectionStatus from "./components/ConnectionStatus";
import "./App.css";

function App() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const handleConnectionChange = (status) => {
    setConnectionStatus(status);
  };

  return (
    <div className="App">
      <ConnectionStatus isConnected={connectionStatus === 'connected'} />
      <Dashboard onConnectionChange={handleConnectionChange} />
    </div>
  );
}

export default App;