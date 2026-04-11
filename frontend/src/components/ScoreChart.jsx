import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { FaChartLine, FaInfoCircle } from "react-icons/fa";

export default function ScoreChart({ alerts }) {
  const data = alerts.map(a => ({
    time: new Date(a.created_at).toLocaleTimeString(),
    score: a.score,
    severity: a.severity
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div
          className="shadow-lg rounded-lg p-3"
          style={{
            backgroundColor: 'var(--color-card-bg)',
            borderColor: 'var(--color-border-light)',
            border: '1px solid',
          }}
        >
          <p className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
            Time: {label}
          </p>
          <p className="text-sm font-medium" style={{ color: 'var(--color-primary)' }}>
            Score: {payload[0].value.toFixed(2)}
          </p>
          <p className="text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
            Severity: {payload[0].payload.severity}
          </p>
        </div>
      );
    }
    return null;
  };

  // Calculate average score for reference
  const avgScore = alerts.length > 0 
    ? (alerts.reduce((sum, a) => sum + a.score, 0) / alerts.length).toFixed(2)
    : 0;

  return (
    <div
      className="rounded-xl shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-300"
      style={{
        backgroundColor: 'var(--color-card-bg)',
        borderColor: 'var(--color-border-light)',
        border: '1px solid',
      }}
    >
      <div
        className="px-6 py-4"
        style={{
          backgroundImage: 'linear-gradient(to right, var(--color-bg-secondary), var(--color-bg-primary))',
          borderBottomColor: 'var(--color-border-light)',
          borderBottom: '1px solid',
        }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FaChartLine className="text-xl" style={{ color: 'var(--color-primary)' }} />
            <h3 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>
              Anomaly Score Trend
            </h3>
          </div>
          <div className="flex items-center space-x-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            <FaInfoCircle className="text-xs" />
            <span>Average Score: {avgScore}</span>
          </div>
        </div>
        <p className="text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
          Real-time anomaly detection scores over time
        </p>
      </div>
      
      <div className="p-4" style={{ height: 350 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-light)" />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 12, fill: 'var(--color-text-secondary)' }}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border-light)' }}
            />
            <YAxis
              label={{
                value: 'Anomaly Score',
                angle: -90,
                position: 'insideLeft',
                style: { fill: 'var(--color-text-secondary)', fontSize: 12 },
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              height={36}
              formatter={(value) => (
                <span style={{ fontSize: '14px', color: 'var(--color-text-secondary)' }}>
                  Anomaly Score
                </span>
              )}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="var(--color-primary)"
              strokeWidth={2}
              dot={{
                fill: 'var(--color-primary)',
                stroke: '#fff',
                strokeWidth: 2,
                r: 4,
              }}
              activeDot={{
                fill: 'var(--color-primary)',
                stroke: '#fff',
                strokeWidth: 2,
                r: 6,
              }}
              name="Score"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend/Insights */}
      {alerts.length > 0 && (
        <div
          className="px-6 py-3"
          style={{
            backgroundColor: 'var(--color-bg-tertiary)',
            borderTopColor: 'var(--color-border-light)',
            borderTop: '1px solid',
          }}
        >
          <div className="flex items-center justify-between text-xs" style={{ color: 'var(--color-text-secondary)' }}>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: 'var(--color-primary)' }}
                ></div>
                <span>High Risk: &gt;70</span>
              </div>
              <div className="flex items-center space-x-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: '#f59e0b' }}
                ></div>
                <span>Medium Risk: 30-70</span>
              </div>
              <div className="flex items-center space-x-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: '#22c55e' }}
                ></div>
                <span>Low Risk: &lt;30</span>
              </div>
            </div>
            <div style={{ opacity: 0.6 }}>
              Last {alerts.length} alerts
            </div>
          </div>
        </div>
      )}
    </div>
  );
}