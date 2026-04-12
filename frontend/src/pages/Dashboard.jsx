import { useEffect, useState } from "react";
import API from "../services/api";
import AlertsTable from "../components/AlertsTable";
import FeedbackForm from "../components/FeedbackForm";
import DashboardCards from "../components/DashboardCards";
import ScoreChart from "../components/ScoreChart";
import { FaBell, FaChartLine, FaCheckCircle, FaExclamationTriangle } from "react-icons/fa";
import { FiRefreshCw } from "react-icons/fi";

export default function Dashboard({ user, token }) {
  const isAnalyst = user?.role === "analyst";
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchAlerts = async () => {
    setIsLoading(true);
    try {
      const res = await API.get("/alerts");
      setAlerts(res.data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Failed to fetch alerts:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualRefresh = () => {
    fetchAlerts();
  };

  useEffect(() => {
    fetchAlerts();
    let interval;
    
    if (autoRefresh) {
      interval = setInterval(fetchAlerts, 5000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  // Calculate stats for header
  const totalAlerts = alerts.length;
  const criticalAlerts = alerts.filter(a => a.severity === "critical").length;
  const resolvedAlerts = alerts.filter(a => a.status === "resolved").length;
  const avgScore = alerts.length > 0 
    ? (alerts.reduce((sum, a) => sum + (a.score || 0), 0) / alerts.length).toFixed(1)
    : 0;

  return (
    <div 
      className="min-h-screen transition-colors duration-200"
      style={{
        backgroundColor: 'var(--color-bg-secondary)',
      }}
    >
      {/* Header Section */}
      <div 
        className="border-b transition-colors duration-200 sticky top-16 z-40"
        style={{
          backgroundColor: 'var(--color-bg-primary)',
          borderColor: 'var(--color-border-primary)',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
            <div>
              <h1 
                className="text-2xl font-bold flex items-center space-x-2 transition-colors duration-200"
                style={{ color: 'var(--color-text-primary)' }}
              >
                <FaChartLine style={{ color: 'var(--color-primary)' }} />
                <span>Dashboard</span>
              </h1>
              <p 
                className="text-sm mt-1 transition-colors duration-200"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                Monitor and manage system alerts
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Stats Summary */}
              <div className="hidden md:flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  <FaBell style={{ color: 'var(--color-primary)' }} />
                  <span 
                    className="font-semibold transition-colors duration-200"
                    style={{ color: 'var(--color-text-primary)' }}
                  >
                    {totalAlerts}
                  </span>
                  <span 
                    className="transition-colors duration-200"
                    style={{ color: 'var(--color-text-secondary)' }}
                  >
                    Total
                  </span>
                </div>
                <div className="flex items-center space-x-1">
                  <FaExclamationTriangle className="text-red-500" />
                  <span 
                    className="font-semibold transition-colors duration-200"
                    style={{ color: 'var(--color-text-primary)' }}
                  >
                    {criticalAlerts}
                  </span>
                  <span 
                    className="transition-colors duration-200"
                    style={{ color: 'var(--color-text-secondary)' }}
                  >
                    Critical
                  </span>
                </div>
                <div className="flex items-center space-x-1">
                  <FaCheckCircle className="text-green-500" />
                  <span 
                    className="font-semibold transition-colors duration-200"
                    style={{ color: 'var(--color-text-primary)' }}
                  >
                    {resolvedAlerts}
                  </span>
                  <span 
                    className="transition-colors duration-200"
                    style={{ color: 'var(--color-text-secondary)' }}
                  >
                    Resolved
                  </span>
                </div>
              </div>
              
              {/* Refresh Controls */}
              <div className="flex items-center space-x-2">
                <label 
                  className="flex items-center space-x-2 text-sm transition-colors duration-200"
                  style={{ color: 'var(--color-text-secondary)' }}
                >
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    style={{
                      borderColor: 'var(--color-border-primary)',
                      backgroundColor: 'var(--color-input-bg)',
                      accentColor: 'var(--color-primary)',
                    }}
                    className="rounded"
                  />
                  <span>Auto-refresh</span>
                </label>
                <button
                  onClick={handleManualRefresh}
                  disabled={isLoading}
                  style={{ color: 'var(--color-text-secondary)' }}
                  className="p-2 transition-colors disabled:opacity-50"
                  onMouseEnter={(e) => {
                    if (!isLoading) e.currentTarget.style.color = 'var(--color-primary)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isLoading) e.currentTarget.style.color = 'var(--color-text-secondary)';
                  }}
                >
                  <FiRefreshCw className={`text-lg ${isLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>
            </div>
          </div>
          
          {/* Last Updated */}
          <div 
            className="text-right text-xs mt-2 transition-colors duration-200"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Loading Overlay */}
        {isLoading && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center md:hidden"
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.3)',
            }}
          >
            <div
              className="rounded-lg p-4 flex items-center space-x-3 transition-colors duration-200"
              style={{
                backgroundColor: 'var(--color-card-bg)',
              }}
            >
              <FiRefreshCw className="animate-spin text-xl" style={{ color: 'var(--color-primary)' }} />
              <span style={{ color: 'var(--color-text-primary)' }}>Updating...</span>
            </div>
          </div>
        )}

        {/* Dashboard Cards */}
        <div className="mb-8 animate-fadeInUp">
          <DashboardCards alerts={alerts} />
        </div>

        {/* Score Chart */}
        <div className="mb-8 animate-fadeInUp animation-delay-200">
            <ScoreChart alerts={alerts} />

        </div>

        {/* Alerts Table */}
        <div className={"grid " + (isAnalyst ? "lg:grid-cols-3" : "lg:grid-cols-1") + " gap-8 animate-fadeInUp animation-delay-400"}>
          <div className={isAnalyst ? "lg:col-span-2" : "lg:col-span-1"}>
            <div
              className="rounded-xl shadow-sm overflow-hidden transition-colors duration-200"
              style={{
                backgroundColor: 'var(--color-card-bg)',
                borderColor: 'var(--color-border-light)',
                border: '1px solid',
              }}
            >
              <div
                className="px-6 py-4 transition-colors duration-200"
                style={{
                  borderBottomColor: 'var(--color-border-light)',
                  borderBottom: '1px solid',
                }}
              >
                <h2
                  className="text-lg font-semibold flex items-center space-x-2 transition-colors duration-200"
                  style={{ color: 'var(--color-text-primary)' }}
                >
                  <FaBell style={{ color: 'var(--color-primary)' }} />
                  <span>Alerts ({alerts.length})</span>
                </h2>
              </div>
              <AlertsTable alerts={alerts} onSelect={isAnalyst ? setSelected : undefined} />
            </div>
          </div>

          {isAnalyst && (
            <div className="lg:col-span-1">
              <div className="sticky top-28">
                <div
                  className="rounded-xl shadow-sm overflow-hidden transition-colors duration-200"
                  style={{
                    backgroundColor: 'var(--color-card-bg)',
                    borderColor: 'var(--color-border-light)',
                    border: '1px solid',
                  }}
                >
                  <div
                    className="px-6 py-4"
                    style={{
                      backgroundImage: 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%)',
                      borderBottomColor: 'var(--color-border-light)',
                      borderBottom: '1px solid',
                    }}
                  >
                    <h2 className="text-lg font-semibold text-white flex items-center space-x-2">
                      <FaCheckCircle />
                      <span>Feedback Form</span>
                    </h2>
                    <p className="text-white text-xs opacity-90 mt-1">
                      {selected ? "Selected alert: " + selected.id : "Select an alert to provide feedback"}
                    </p>
                  </div>
                  <FeedbackForm alert={selected} onSuccess={fetchAlerts} token={token} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeInUp {
          animation: fadeInUp 0.5s ease-out;
        }
        .animation-delay-200 {
          animation-delay: 0.2s;
          opacity: 0;
          animation-fill-mode: forwards;
        }
        .animation-delay-400 {
          animation-delay: 0.4s;
          opacity: 0;
          animation-fill-mode: forwards;
        }
      `}</style>
    </div>
  );
}