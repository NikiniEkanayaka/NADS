import React, { useState } from "react";
import axios from "axios";

export default function AddUser({ token }) {
  const [form, setForm] = useState({
    name: "",
    username: "",
    password: "",
    role: "analyst",
  });
  const [message, setMessage] = useState("");

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://localhost:8000/auth/signup", form, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setMessage(res.data.message);
      setForm({ name: "", username: "", password: "", role: "analyst" });
    } catch (err) {
      setMessage(err.response?.data?.detail || "Error creating user");
    }
  };

  return (
    <div
      className="max-w-md mx-auto mt-10 p-6 rounded shadow"
      style={{
        backgroundColor: 'var(--color-card-bg)',
        borderColor: 'var(--color-border-light)',
        border: '1px solid',
      }}
    >
      <h2 className="text-2xl mb-4 font-bold" style={{ color: 'var(--color-text-primary)' }}>
        Add New User
      </h2>
      {message && (
        <p className="mb-2" style={{ color: 'var(--color-success)' }}>
          {message}
        </p>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          name="name"
          value={form.name}
          onChange={handleChange}
          placeholder="Full Name"
          className="w-full p-2 rounded transition-colors"
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
          required
        />
        <input
          name="username"
          value={form.username}
          onChange={handleChange}
          placeholder="Username"
          className="w-full p-2 rounded transition-colors"
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
          required
        />
        <input
          name="password"
          value={form.password}
          onChange={handleChange}
          type="password"
          placeholder="Password"
          className="w-full p-2 rounded transition-colors"
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
          required
        />
        <select
          name="role"
          value={form.role}
          onChange={handleChange}
          className="w-full p-2 rounded transition-colors"
          style={{
            backgroundColor: 'var(--color-input-bg)',
            borderColor: 'var(--color-input-border)',
            border: '1px solid',
            color: 'var(--color-input-text)',
          }}
        >
          <option value="analyst">Analyst</option>
          <option value="admin">Admin</option>
        </select>
        <button
          type="submit"
          className="w-full text-white py-2 rounded transition-all duration-200"
          style={{
            backgroundColor: 'var(--color-primary)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.opacity = '0.9';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.opacity = '1';
          }}
        >
          Add User
        </button>
      </form>
    </div>
  );
}