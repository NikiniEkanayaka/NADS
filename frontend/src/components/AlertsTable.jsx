import React, { useState } from "react";
import { FaEye, FaCheckCircle, FaExclamationTriangle, FaInfoCircle, FaSort, FaSortUp, FaSortDown } from "react-icons/fa";

export default function AlertsTable({ alerts, onSelect }) {
  const [sortField, setSortField] = useState("created_at");
  const showActions = typeof onSelect === "function";
  const [sortDirection, setSortDirection] = useState("desc");
  const [hoveredRow, setHoveredRow] = useState(null);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const getSortIcon = (field) => {
    if (sortField !== field) return <FaSort style={{ color: 'var(--color-text-secondary)' }} />;
    return sortDirection === "asc" ? 
      <FaSortUp style={{ color: 'var(--color-primary)' }} /> : 
      <FaSortDown style={{ color: 'var(--color-primary)' }} />;
  };

  const sortedAlerts = [...alerts].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];
    
    if (sortField === "created_at") {
      aVal = new Date(aVal);
      bVal = new Date(bVal);
    }
    
    if (aVal < bVal) return sortDirection === "asc" ? -1 : 1;
    if (aVal > bVal) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  const getSeverityColor = (severity) => {
    switch(severity?.toLowerCase()) {
      case "critical":
      case "high":
        return "bg-red-100 text-red-700 border-red-200";
      case "medium":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "low":
        return "bg-green-100 text-green-700 border-green-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const getScoreColor = (score) => {
    if (score >= 70) return "text-red-600 font-bold";
    if (score >= 30) return "text-yellow-600 font-semibold";
    return "text-green-600";
  };

  if (alerts.length === 0) {
    return (
      <div 
        className="rounded-xl shadow-sm border p-8 text-center transition-colors duration-200"
        style={{
          backgroundColor: 'var(--color-card-bg)',
          borderColor: 'var(--color-card-border)',
        }}
      >
        <div 
          className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-colors duration-200"
          style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
        >
          <FaInfoCircle 
            className="text-2xl"
            style={{ color: 'var(--color-text-secondary)' }}
          />
        </div>
        <p 
          className="transition-colors duration-200"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          No alerts found
        </p>
        <p 
          className="text-sm mt-1 transition-colors duration-200"
          style={{ color: 'var(--color-text-tertiary)' }}
        >
          Alerts will appear here as they are detected
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead 
          className="border-b transition-colors duration-200"
          style={{
            backgroundColor: 'var(--color-bg-tertiary)',
            borderColor: 'var(--color-border-primary)',
          }}
        >
          <tr>
            <th 
              className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider cursor-pointer hover:opacity-80 transition-opacity"
              style={{ color: 'var(--color-text-secondary)' }}
              onClick={() => handleSort("created_at")}
            >
              <div className="flex items-center space-x-1">
                <span>Time</span>
                {getSortIcon("created_at")}
              </div>
            </th>
            <th 
              className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider cursor-pointer hover:opacity-80 transition-opacity"
              style={{ color: 'var(--color-text-secondary)' }}
              onClick={() => handleSort("score")}
            >
              <div className="flex items-center space-x-1">
                <span>Score</span>
                {getSortIcon("score")}
              </div>
            </th>
            <th 
              className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider cursor-pointer hover:opacity-80 transition-opacity"
              style={{ color: 'var(--color-text-secondary)' }}
              onClick={() => handleSort("severity")}
            >
              <div className="flex items-center space-x-1">
                <span>Severity</span>
                {getSortIcon("severity")}
              </div>
            </th>
            <th 
              className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider transition-colors duration-200"
              style={{ color: 'var(--color-text-secondary)' }}
            >
              Status
            </th>
            {showActions && (
              <th 
                className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider transition-colors duration-200"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                Action
              </th>
            )}
          </tr>
        </thead>
        <tbody 
          className="transition-colors duration-200"
          style={{ borderColor: 'var(--color-border-primary)' }}
        >
          {sortedAlerts.map((alert, index) => (
            <tr 
              key={alert.id}
              className="transition-all duration-200 border-b"
              style={{
                borderColor: 'var(--color-border-light)',
                backgroundColor: hoveredRow === alert.id ? 'rgba(28, 155, 201, 0.05)' : 'transparent',
              }}
              onMouseEnter={() => setHoveredRow(alert.id)}
              onMouseLeave={() => setHoveredRow(null)}
            >
              <td 
                className="px-4 py-3 text-sm transition-colors duration-200"
                style={{ color: 'var(--color-text-primary)' }}
              >
                {new Date(alert.created_at).toLocaleString()}
              </td>
              <td className="px-4 py-3">
                <span className={`text-sm font-medium ${getScoreColor(alert.score)}`}>
                  {alert.score?.toFixed(2)}
                </span>
              </td>
              <td className="px-4 py-3">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getSeverityColor(alert.severity)}`}>
                  <FaExclamationTriangle className="mr-1 text-xs" />
                  {alert.severity?.toUpperCase() || "UNKNOWN"}
                </span>
              </td>
              <td className="px-4 py-3">
                <span
                  className="inline-flex items-center text-xs transition-colors duration-200"
                  style={{
                    color: 'var(--color-success)',
                  }}
                >
                  <FaCheckCircle className="mr-1 text-xs" />
                  Active
                </span>
              </td>
              {showActions && (
                <td className="px-4 py-3">
                  <button
                    onClick={() => onSelect(alert)}
                    style={{
                      backgroundColor: 'var(--color-primary)',
                    }}
                    className="inline-flex items-center px-3 py-1.5 text-white text-sm rounded-lg transition-all duration-200 shadow-sm hover:shadow-md transform hover:scale-105"
                  >
                    <FaEye className="mr-1 text-xs" />
                    Review
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Footer with stats */}
      <div 
        className="px-4 py-3 border-t transition-colors duration-200"
        style={{
          backgroundColor: 'var(--color-bg-tertiary)',
          borderColor: 'var(--color-border-primary)',
        }}
      >
        <div 
          className="flex items-center justify-between text-xs transition-colors duration-200"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          <div className="flex items-center space-x-4">
            <span>Total: {alerts.length} alerts</span>
            <span>Showing: {sortedAlerts.length} sorted</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="inline-block w-2 h-2 bg-green-500 rounded-full"></span>
            <span>Low Risk</span>
            <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full ml-2"></span>
            <span>Medium Risk</span>
            <span className="inline-block w-2 h-2 bg-red-500 rounded-full ml-2"></span>
            <span>High Risk</span>
          </div>
        </div>
      </div>
    </div>
  );
}