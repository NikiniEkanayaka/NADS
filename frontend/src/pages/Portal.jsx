// src/pages/Portal.jsx
import React, { useState } from "react";
import Navbar from "../components/Navbar";
import AddUser from "./AddUser"; // admin-only
import Dashboard from "./Dashboard"; // alerts + feedback
import Profile from "./Profile";

export default function Portal({ user, token, onLogout }) {
  const [currentTab, setCurrentTab] = useState("dashboard");

  const renderTab = () => {
    switch (currentTab) {
      case "dashboard":
        return <Dashboard user={user} token={token} />;
      case "profile":
        return <Profile user={user} token={token} onLogout={onLogout} />;
      case "adduser":
        if (user.role !== "admin") return <p className="p-6">Access Denied</p>;
        return <AddUser token={token} />;
      default:
        return <Dashboard token={token} />;
    }
  };

  return (
    <div>
      <Navbar
        user={user}
        onLogout={onLogout}
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
      />
      <div className="p-6">{renderTab()}</div>
    </div>
  );
}