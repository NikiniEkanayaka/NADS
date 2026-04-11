import { useEffect, useState } from "react";
import API from "../services/api";
import AlertsTable from "../components/AlertsTable";
import FeedbackForm from "../components/FeedbackForm";
import DashboardCards from "../components/DashboardCards";
import ScoreChart from "../components/ScoreChart";

export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);

  const fetchAlerts = async () => {
    const res = await API.get("/alerts");
    setAlerts(res.data);
  };

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <DashboardCards alerts={alerts} />
      <ScoreChart alerts={alerts} />
      <AlertsTable alerts={alerts} onSelect={setSelected} />
      <FeedbackForm alert={selected} />
    </div>
  );
}