import React, { useState } from "react";
import API from "../services/api";
import { FaCheckCircle, FaTimesCircle, FaSpinner, FaComment, FaStar, FaStarHalf } from "react-icons/fa";

export default function FeedbackForm({ alert, token, onSuccess }) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("success");
  const [confidence, setConfidence] = useState(50);

  if (!alert) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center">
        <div
          className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
          style={{
            backgroundColor: 'var(--color-bg-tertiary)',
          }}
        >
          <FaComment className="text-2xl" style={{ color: 'var(--color-text-secondary)' }} />
        </div>
        <p style={{ color: 'var(--color-text-secondary)', }} className="text-sm">
          Select an alert to provide feedback
        </p>
        <p style={{ color: 'var(--color-text-secondary)', marginTop: '4px' }} className="text-xs opacity-75">
          Click on any alert row to review it
        </p>
      </div>
    );
  }

  const handleFeedback = async (label) => {
    setLoading(true);
    setMessage("");
    try {
      await API.post(
        "/feedback",
        { 
          alert_id: alert.id, 
          label, 
          analyst: "current_user",
          confidence: confidence
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setMessageType("success");
      setMessage("Feedback saved successfully!");
      if (onSuccess) onSuccess();
      setTimeout(() => setMessage(""), 3000);
    } catch (err) {
      setMessageType("error");
      setMessage(err.response?.data?.detail || "Error saving feedback");
      setTimeout(() => setMessage(""), 3000);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      {/* Alert Info */}
      <div
        className="mb-6 p-4 rounded-lg"
        style={{
          backgroundColor: 'var(--color-bg-tertiary)',
          borderColor: 'var(--color-border-light)',
          border: '1px solid',
        }}
      >
        <div className="flex justify-between items-start mb-2">
          <h4 className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
            Alert #{alert.id}
          </h4>
          <span
            className="px-2 py-1 rounded-full text-xs font-medium"
            style={{
              backgroundColor: 
                alert.severity === "critical" || alert.severity === "high"
                  ? 'rgba(239, 68, 68, 0.2)'
                  : alert.severity === "medium"
                  ? 'rgba(234, 179, 8, 0.2)'
                  : 'rgba(34, 197, 94, 0.2)',
              color:
                alert.severity === "critical" || alert.severity === "high"
                  ? '#ef4444'
                  : alert.severity === "medium"
                  ? '#eab308'
                  : '#22c55e',
            }}
          >
            {alert.severity?.toUpperCase() || "UNKNOWN"}
          </span>
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span style={{ color: 'var(--color-text-secondary)' }}>Anomaly Score:</span>
            <span className="font-medium" style={{ color: 'var(--color-text-primary)' }}>
              {alert.score?.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span style={{ color: 'var(--color-text-secondary)' }}>Detected:</span>
            <span style={{ color: 'var(--color-text-primary)' }}>
              {new Date(alert.created_at).toLocaleString()}
            </span>
          </div>
          {alert.description && (
            <div
              className="mt-2 pt-2"
              style={{
                borderTop: '1px solid',
                borderColor: 'var(--color-border-light)',
              }}
            >
              <span style={{ color: 'var(--color-text-secondary)' }} className="block mb-1">
                Description:
              </span>
              <p style={{ color: 'var(--color-text-primary)' }} className="text-sm">
                {alert.description}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Confidence Slider */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
          Confidence Level
        </label>
        <div className="flex items-center space-x-3">
          <input
            type="range"
            min="0"
            max="100"
            value={confidence}
            onChange={(e) => setConfidence(parseInt(e.target.value))}
            className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, var(--color-primary) 0%, var(--color-primary) ${confidence}%, var(--color-border-light) ${confidence}%, var(--color-border-light) 100%)`
            }}
          />
          <div className="flex items-center space-x-1 min-w-[60px]">
            {confidence >= 80 ? (
              <FaStar style={{ color: '#f59e0b' }} />
            ) : confidence >= 50 ? (
              <FaStarHalf style={{ color: '#f59e0b' }} />
            ) : (
              <FaStar style={{ color: 'var(--color-text-secondary)' }} />
            )}
            <span className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
              {confidence}%
            </span>
          </div>
        </div>
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
          <span>Low Confidence</span>
          <span>Medium</span>
          <span>High Confidence</span>
        </div>
      </div>

      {/* Message Notification */}
      {message && (
        <div
          className="mb-4 p-3 rounded-lg flex items-center space-x-2"
          style={{
            backgroundColor:
              messageType === "success"
                ? 'rgba(34, 197, 94, 0.2)'
                : 'rgba(239, 68, 68, 0.2)',
            color: messageType === "success" ? '#22c55e' : '#ef4444',
            borderColor: messageType === "success" ? '#22c55e' : '#ef4444',
            border: '1px solid',
          }}
        >
          {messageType === "success" ? (
            <FaCheckCircle style={{ color: '#22c55e' }} />
          ) : (
            <FaTimesCircle style={{ color: '#ef4444' }} />
          )}
          <span className="text-sm">{message}</span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-3">
        <button
          disabled={loading}
          onClick={() => handleFeedback(true)}
          className="w-full text-white px-4 py-2.5 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-sm"
          style={{
            background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
          }}
          onMouseEnter={(e) => {
            if (!loading) e.currentTarget.style.opacity = '0.9';
          }}
          onMouseLeave={(e) => {
            if (!loading) e.currentTarget.style.opacity = '1';
          }}
        >
          {loading ? (
            <FaSpinner className="animate-spin" />
          ) : (
            <FaCheckCircle />
          )}
          <span>True Positive</span>
        </button>

        <button
          disabled={loading}
          onClick={() => handleFeedback(false)}
          className="w-full text-white px-4 py-2.5 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-sm"
          style={{
            background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
          }}
          onMouseEnter={(e) => {
            if (!loading) e.currentTarget.style.opacity = '0.9';
          }}
          onMouseLeave={(e) => {
            if (!loading) e.currentTarget.style.opacity = '1';
          }}
        >
          {loading ? (
            <FaSpinner className="animate-spin" />
          ) : (
            <FaTimesCircle />
          )}
          <span>False Positive</span>
        </button>
      </div>

      {/* Help Text */}
      <div className="mt-4 text-xs text-center" style={{ color: 'var(--color-text-secondary)', opacity: 0.5 }}>
        Your feedback helps improve detection accuracy
      </div>
    </div>
  );
}