export default function DashboardCards({ alerts }) {
  const total = alerts.length;
  const high = alerts.filter(a => a.severity === "high").length;

  return (
    <div className="grid grid-cols-3 gap-4 mb-6">
      <Card title="Total Alerts" value={total} />
      <Card title="High Severity" value={high} />
      <Card title="Precision (mock)" value="92%" />
    </div>
  );
}

function Card({ title, value }) {
  return (
    <div 
      className="p-4 rounded-lg shadow-sm border transition-all duration-200"
      style={{
        backgroundColor: 'var(--color-card-bg)',
        borderColor: 'var(--color-card-border)',
      }}
    >
      <h3 
        className="text-sm transition-colors duration-200"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        {title}
      </h3>
      <p 
        className="text-2xl font-bold mt-2 transition-colors duration-200"
        style={{ color: 'var(--color-text-primary)' }}
      >
        {value}
      </p>
    </div>
  );
}