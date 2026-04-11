import React, { useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { themeList } from "../config/themes";
import { FaPalette } from "react-icons/fa";

export default function ThemeSwitcher() {
  const { theme, setThemeTo } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg transition-colors duration-200"
        style={{
          color: 'var(--color-text-secondary)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--color-bg-hover)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent';
        }}
        title="Change theme"
      >
        <FaPalette className="text-lg" />
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-48 rounded-lg shadow-lg py-2 z-50 animate-fadeInUp"
          style={{
            backgroundColor: 'var(--color-card-bg)',
            borderColor: 'var(--color-border-light)',
            border: '1px solid',
          }}
        >
          <div className="px-4 py-2 text-xs font-semibold uppercase" style={{ color: 'var(--color-text-secondary)' }}>
            Themes
          </div>

          {themeList.map((t) => (
            <button
              key={t.id}
              onClick={() => {
                setThemeTo(t.id);
                setIsOpen(false);
              }}
              className="w-full text-left px-4 py-2 text-sm transition-colors duration-150 flex items-center space-x-2"
              style={{
                color: theme === t.id ? 'var(--color-primary)' : 'var(--color-text-primary)',
                backgroundColor: theme === t.id ? 'var(--color-bg-tertiary)' : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (theme !== t.id) {
                  e.currentTarget.style.backgroundColor = 'var(--color-bg-hover)';
                }
              }}
              onMouseLeave={(e) => {
                if (theme !== t.id) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{
                  backgroundColor:
                    t.id === "light"
                      ? "#ffffff"
                      : t.id === "dark"
                      ? "#1e293b"
                      : t.id === "ocean"
                      ? "#0369a1"
                      : t.id === "forest"
                      ? "#15803d"
                      : "#ea580c",
                  border: theme === t.id ? '2px solid var(--color-primary)' : 'none',
                  boxSizing: 'border-box',
                }}
              />
              <span>{t.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
