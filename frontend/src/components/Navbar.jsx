import React, { useState } from "react";
import { FaUserCircle, FaSignOutAlt, FaTachometerAlt, FaUserPlus, FaChevronDown, FaUser } from "react-icons/fa";
import ThemeSwitcher from "./ThemeSwitcher";
import { useTheme } from "../context/ThemeContext";

export default function Navbar({ user, onLogout, currentTab, setCurrentTab }) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { theme } = useTheme();
  
  const tabs = [
    { name: "Dashboard", key: "dashboard", icon: FaTachometerAlt },
    { name: "Add User", key: "adduser", icon: FaUserPlus, adminOnly: true },
  ];

  return (
    <nav 
      className="shadow-lg sticky top-0 z-50 transition-colors duration-200"
      style={{
        backgroundColor: 'var(--color-navbar-bg)',
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="h-8 w-8 rounded-lg flex items-center justify-center"
                style={{
                  backgroundImage: 'linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%)',
                }}>
                <span className="text-white font-bold text-lg">N</span>
              </div>
              <span 
                className="ml-3 text-xl font-bold transition-colors duration-200"
                style={{ color: 'var(--color-navbar-text)' }}
              >
                NADS
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {tabs.map((tab) => (
              (!tab.adminOnly || user.role === "admin") && (
                <button
                  key={tab.key}
                  onClick={() => setCurrentTab(tab.key)}
                  className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center space-x-2 shadow-md"
                  style={{
                    backgroundColor: currentTab === tab.key ? 'var(--color-primary)' : 'transparent',
                    color: currentTab === tab.key ? 'white' : 'var(--color-navbar-text)',
                    opacity: currentTab === tab.key ? 1 : 0.7,
                  }}
                  onMouseEnter={(e) => {
                    if (currentTab !== tab.key) {
                      e.currentTarget.style.opacity = '1';
                      e.currentTarget.style.backgroundColor = 'var(--color-bg-hover)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (currentTab !== tab.key) {
                      e.currentTarget.style.opacity = '0.7';
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <tab.icon className="text-sm" />
                  <span>{tab.name}</span>
                </button>
              )
            ))}
          </div>

          {/* User Menu */}
          <div className="flex items-center space-x-4">
            <ThemeSwitcher />
            <div className="relative">
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center space-x-3 focus:outline-none group"
              >
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <FaUserCircle
                      className="text-2xl transition-colors"
                      style={{
                        color: 'var(--color-text-secondary)',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = 'var(--color-primary)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = 'var(--color-text-secondary)';
                      }}
                    />
                    {user.role === "admin" && (
                      <span
                        className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2"
                        style={{
                          backgroundColor: '#10b981',
                          borderColor: 'var(--color-bg-primary)',
                        }}
                      ></span>
                    )}
                  </div>
                  <div className="hidden md:block text-left">
                    <p
                      className="text-sm font-medium"
                      style={{ color: 'var(--color-text-primary)' }}
                    >
                      {user.username}
                    </p>
                    <p
                      className="text-xs capitalize"
                      style={{ color: 'var(--color-text-secondary)' }}
                    >
                      {user.role}
                    </p>
                  </div>
                  <FaChevronDown
                    className="text-xs transition-transform duration-200"
                    style={{
                      color: 'var(--color-text-secondary)',
                      transform: isDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                    }}
                  />
                </div>
              </button>

              {/* Dropdown Menu */}
              {isDropdownOpen && (
                <div
                  className="absolute right-0 mt-2 w-48 rounded-lg shadow-lg py-1 animate-fadeInUp"
                  style={{
                    backgroundColor: 'var(--color-card-bg)',
                    borderColor: 'var(--color-card-border)',
                    border: '1px solid',
                  }}
                >
                  <div
                    className="px-4 py-2 border-b md:hidden"
                    style={{
                      borderColor: 'var(--color-border-light)',
                    }}
                  >
                    <p
                      className="text-sm font-medium"
                      style={{ color: 'var(--color-text-primary)' }}
                    >
                      {user.username}
                    </p>
                    <p
                      className="text-xs capitalize"
                      style={{ color: 'var(--color-text-secondary)' }}
                    >
                      {user.role}
                    </p>
                  </div>

                  <button
                    onClick={() => {
                      setCurrentTab("profile");
                      setIsDropdownOpen(false);
                    }}
                    className="w-full text-left px-4 py-2 text-sm flex items-center space-x-2 transition-colors"
                    style={{
                      color: 'var(--color-text-primary)',
                      backgroundColor: 'transparent',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'var(--color-bg-hover)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <FaUser />
                    <span>Profile</span>
                  </button>

                  <button
                    onClick={onLogout}
                    className="w-full text-left px-4 py-2 text-sm flex items-center space-x-2 transition-colors"
                    style={{
                      color: 'var(--color-danger)',
                      backgroundColor: 'transparent',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <FaSignOutAlt />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div
        className="md:hidden border-t"
        style={{
          borderColor: 'var(--color-border-light)',
          backgroundColor: 'var(--color-bg-secondary)',
        }}
      >
        <div className="flex justify-around py-2">
          {tabs.map((tab) => (
            (!tab.adminOnly || user.role === "admin") && (
              <button
                key={tab.key}
                onClick={() => setCurrentTab(tab.key)}
                className="flex flex-col items-center px-3 py-2 rounded-lg transition-all duration-200"
                style={{
                  color: currentTab === tab.key ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                }}
                onMouseEnter={(e) => {
                  if (currentTab !== tab.key) {
                    e.currentTarget.style.color = 'var(--color-text-primary)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentTab !== tab.key) {
                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                  }
                }}
              >
                <tab.icon className="text-lg" />
                <span className="text-xs mt-1">{tab.name}</span>
              </button>
            )
          ))}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeInUp {
          animation: fadeInUp 0.2s ease-out;
        }
      `}</style>
    </nav>
  );
}