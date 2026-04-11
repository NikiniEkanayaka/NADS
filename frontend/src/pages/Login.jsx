import React, { useState } from "react";
import axios from "axios";
import { FaUser, FaLock, FaEye, FaEyeSlash, FaSpinner } from "react-icons/fa";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [usernameFocus, setUsernameFocus] = useState(false);
  const [passwordFocus, setPasswordFocus] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    
    try {
      const res = await axios.post("http://localhost:8000/auth/login", {
        username,
        password,
      });
      onLogin(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
      // Shake animation for error
      const form = e.target;
      form.classList.add("shake");
      setTimeout(() => form.classList.remove("shake"), 500);
    } finally {
      setIsLoading(false);
    }
  };

  // Add this CSS to your global styles or component
  const styles = `
    @keyframes shake {
      0%, 100% { transform: translateX(0); }
      25% { transform: translateX(-5px); }
      75% { transform: translateX(5px); }
    }
    .shake {
      animation: shake 0.3s ease-in-out;
    }
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
    .fade-in-up {
      animation: fadeInUp 0.5s ease-out;
    }
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }
    .pulse {
      animation: pulse 2s ease-in-out infinite;
    }
  `;

  return (
    <>
      <style>{styles}</style>
      <div
        className="min-h-screen flex items-center justify-center p-4 transition-colors duration-200"
        style={{
          backgroundImage: 'linear-gradient(to bottom right, var(--color-bg-primary), var(--color-bg-secondary))',
        }}
      >
        <div className="fade-in-up w-full max-w-md">
          {/* Brand/Logo Section */}
          <div className="text-center mb-8">
            <div
              className="inline-flex items-center justify-center w-20 h-20 rounded-2xl shadow-lg mb-4 transform transition-transform hover:scale-105"
              style={{ backgroundColor: 'var(--color-primary)' }}
            >
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold transition-colors duration-200" style={{ color: 'var(--color-text-primary)' }}>
              Welcome Back
            </h1>
            <p className="mt-2 transition-colors duration-200" style={{ color: 'var(--color-text-secondary)' }}>
              Sign in to your account
            </p>
          </div>

          {/* Login Card */}
          <div
            className="rounded-2xl shadow-xl p-8 transition-colors duration-200"
            style={{ backgroundColor: 'var(--color-card-bg)' }}
          >
            {error && (
              <div
                className="mb-6 p-4 rounded-lg transition-colors duration-200"
                style={{
                  backgroundColor: 'rgba(239, 68, 68, 0.1)',
                  borderLeft: '4px solid var(--color-danger)',
                }}
              >
                <p style={{ color: 'var(--color-danger)' }} className="text-sm">
                  {error}
                </p>
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-6">
              {/* Username Field */}
              <div className="relative">
                <div
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 transition-colors duration-200"
                  style={{
                    color: usernameFocus || username ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                  }}
                >
                  <FaUser />
                </div>
                <input
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onFocus={() => setUsernameFocus(true)}
                  onBlur={() => setUsernameFocus(false)}
                  className="w-full pl-10 pr-4 py-3 border-2 rounded-lg focus:outline-none transition-all duration-200"
                  style={{
                    backgroundColor: 'var(--color-input-bg)',
                    color: 'var(--color-input-text)',
                    borderColor: usernameFocus ? 'var(--color-primary)' : 'var(--color-input-border)',
                    boxShadow: usernameFocus ? 'rgba(0, 0, 0, 0.05) 0 0 0 3px var(--color-primary)' : 'none',
                  }}
                  required
                  disabled={isLoading}
                />
                <label
                  className="absolute left-10 -top-2.5 px-2 text-sm transition-all duration-200"
                  style={{
                    backgroundColor: 'var(--color-card-bg)',
                    color: usernameFocus || username ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                    opacity: usernameFocus || username ? 1 : 0,
                  }}
                >
                  Username
                </label>
              </div>

              {/* Password Field */}
              <div className="relative">
                <div
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 transition-colors duration-200"
                  style={{
                    color: passwordFocus || password ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                  }}
                >
                  <FaLock />
                </div>
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setPasswordFocus(true)}
                  onBlur={() => setPasswordFocus(false)}
                  className="w-full pl-10 pr-12 py-3 border-2 rounded-lg focus:outline-none transition-all duration-200"
                  style={{
                    backgroundColor: 'var(--color-input-bg)',
                    color: 'var(--color-input-text)',
                    borderColor: passwordFocus ? 'var(--color-primary)' : 'var(--color-input-border)',
                    boxShadow: passwordFocus ? 'rgba(0, 0, 0, 0.05) 0 0 0 3px var(--color-primary)' : 'none',
                  }}
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 transition-colors"
                  style={{
                    color: 'var(--color-text-secondary)',
                    cursor: 'pointer',
                  }}
                  disabled={isLoading}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = 'var(--color-text-primary)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                  }}
                >
                  {showPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
                <label
                  className="absolute left-10 -top-2.5 px-2 text-sm transition-all duration-200"
                  style={{
                    backgroundColor: 'var(--color-card-bg)',
                    color: passwordFocus || password ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                    opacity: passwordFocus || password ? 1 : 0,
                  }}
                >
                  Password
                </label>
              </div>

              {/* Additional Options */}
              <div className="flex items-center justify-between text-sm">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    className="w-4 h-4 rounded"
                    style={{
                      borderColor: 'var(--color-input-border)',
                      backgroundColor: 'var(--color-input-bg)',
                      accentColor: 'var(--color-primary)',
                    }}
                  />
                  <span style={{ color: 'var(--color-text-secondary)', transition: 'color 200ms' }}>
                    Remember me
                  </span>
                </label>
                <a
                  href="#"
                  className="transition-colors"
                  style={{
                    color: 'var(--color-primary)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.opacity = '0.8';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.opacity = '1';
                  }}
                >
                  Forgot password?
                </a>
              </div>

              {/* Login Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3 rounded-lg text-white font-semibold transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed disabled:hover:scale-100"
                style={{
                  background: 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%)',
                  boxShadow: 'rgba(0, 0, 0, 0.1) 0 4px 15px',
                }}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <FaSpinner className="animate-spin" />
                    <span>Signing in...</span>
                  </div>
                ) : (
                  "Sign In"
                )}
              </button>

              {/* Sign Up Link */}
              <div className="text-center text-sm transition-colors duration-200" style={{ color: 'var(--color-text-secondary)' }}>
                Don't have an account?{" "}
                <a href="#" className="font-semibold transition-colors" style={{ color: 'var(--color-primary)' }}>
                  Sign up
                </a>
              </div>
            </form>
          </div>

          {/* Footer */}
          <div className="text-center mt-6 text-xs transition-colors duration-200" style={{ color: 'var(--color-text-secondary)' }}>
            By signing in, you agree to our{" "}
            <a href="#" className="transition-colors hover:underline" style={{ color: 'var(--color-primary)' }}>
              Terms of Service
            </a>{" "}
            and{" "}
            <a href="#" className="transition-colors hover:underline" style={{ color: 'var(--color-primary)' }}>
              Privacy Policy
            </a>
          </div>
        </div>
      </div>
    </>
  );
}