import { useState, useEffect } from "react";
import API from "../services/api";

export default function Profile({ user, setUser }) {
  const [form, setForm] = useState({
    name: user.name || "",
    password: "",
  });
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  // Make sure username is set properly in form on mount
  useEffect(() => {
    setForm((prev) => ({
      ...prev,
      name: user.name || "",
    }));
  }, [user]);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const res = await API.put("/auth/profile", form);

      // Use res.data.message and res.data.name
      if (res.status === 200) {
        setMessage(res.data.message || "Profile updated successfully");

        // Update local user state
        setUser((prev) => ({
          ...prev,
          name: res.data.name || prev.name,
          username: prev.username, // keep username
        }));

        setForm({ ...form, password: "" }); // clear password field
      } else {
        setMessage("Update failed");
      }
    } catch (err) {
      console.error(err);
      setMessage(err.response?.data?.detail || "Update failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="max-w-lg mx-auto p-6 rounded-xl shadow"
      style={{
        backgroundColor: 'var(--color-card-bg)',
        borderColor: 'var(--color-border-light)',
        border: '1px solid',
      }}
    >
      <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--color-text-primary)' }}>
        My Profile
      </h2>

      {message && (
        <p
          className="mb-4 font-medium"
          style={{
            color: message.includes("failed") ? 'var(--color-danger)' : 'var(--color-success)',
          }}
        >
          {message}
        </p>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Username (readonly) */}
        <div>
          <label className="block text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            Username
          </label>
          <input
            value={user.username || ""}
            disabled
            className="w-full p-2 rounded"
            style={{
              backgroundColor: 'var(--color-bg-tertiary)',
              borderColor: 'var(--color-border-light)',
              border: '1px solid',
              color: 'var(--color-text-secondary)',
              opacity: 0.6,
            }}
          />
        </div>

        {/* Name */}
        <div>
          <label className="block text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            Name
          </label>
          <input
            name="name"
            value={form.name}
            onChange={handleChange}
            className="w-full p-2 rounded transition-colors"
            placeholder="Enter your name"
            style={{
              backgroundColor: 'var(--color-input-bg)',
              borderColor: 'var(--color-input-border)',
              border: '1px solid',
              color: 'var(--color-input-text)',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'var(--color-primary)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'var(--color-input-border)';
            }}
          />
        </div>

        {/* Password */}
        <div>
          <label className="block text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            New Password
          </label>
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            className="w-full p-2 rounded transition-colors"
            placeholder="Leave blank to keep current"
            style={{
              backgroundColor: 'var(--color-input-bg)',
              borderColor: 'var(--color-input-border)',
              border: '1px solid',
              color: 'var(--color-input-text)',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'var(--color-primary)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'var(--color-input-border)';
            }}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full text-white py-2 rounded transition-all duration-200 disabled:opacity-50"
          style={{
            backgroundColor: 'var(--color-primary)',
          }}
          onMouseEnter={(e) => {
            if (!loading) e.currentTarget.style.opacity = '0.9';
          }}
          onMouseLeave={(e) => {
            if (!loading) e.currentTarget.style.opacity = '1';
          }}
        >
          {loading ? "Updating..." : "Update Profile"}
        </button>
      </form>
    </div>
  );
}