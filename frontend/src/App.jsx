import React, { useState } from "react";
import Login from "./pages/Login";
import Portal from "./pages/Portal";
import { ThemeProvider } from "./context/ThemeContext";

export default function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  const handleLogin = (data) => {
    setUser({ username: data.username || data.sub, role: data.role, name: data.name || "", });
    setToken(data.access_token);

    localStorage.setItem("token", data.access_token);
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem("token");
  };

  return (
    <ThemeProvider>
      {!user ? (
        <Login onLogin={handleLogin} />
      ) : (
        <Portal user={user} token={token} onLogout={handleLogout} setUser={setUser} />
      )}
    </ThemeProvider>
  );
}