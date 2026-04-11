import React, { createContext, useContext, useState, useEffect } from "react";
import { getTheme } from "../config/themes";

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    // Check localStorage for saved theme
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      return savedTheme;
    }
    // Check system preference
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
    return "light";
  });

  useEffect(() => {
    // Update localStorage and DOM
    localStorage.setItem("theme", theme);
    
    const root = document.documentElement;
    const themeConfig = getTheme(theme);
    
    // Update data-theme attribute
    root.setAttribute("data-theme", theme);
    
    // Handle dark class for light/dark mode only
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    
    // Apply CSS variables for theme colors
    const colors = themeConfig.colors;
    
    // Primary colors
    root.style.setProperty("--color-primary", colors.primary);
    root.style.setProperty("--color-primary-dark", colors.primaryDark);
    root.style.setProperty("--color-secondary", colors.secondary);
    root.style.setProperty("--color-success", colors.success);
    root.style.setProperty("--color-warning", colors.warning);
    root.style.setProperty("--color-danger", colors.danger);
    root.style.setProperty("--color-info", colors.info);
    
    // Background colors
    root.style.setProperty("--color-bg-primary", colors.bg.primary);
    root.style.setProperty("--color-bg-secondary", colors.bg.secondary);
    root.style.setProperty("--color-bg-tertiary", colors.bg.tertiary);
    root.style.setProperty("--color-bg-hover", colors.bg.hover);
    
    // Text colors
    root.style.setProperty("--color-text-primary", colors.text.primary);
    root.style.setProperty("--color-text-secondary", colors.text.secondary);
    root.style.setProperty("--color-text-tertiary", colors.text.tertiary);
    root.style.setProperty("--color-text-inverse", colors.text.inverse);
    
    // Border colors
    root.style.setProperty("--color-border-primary", colors.border.primary);
    root.style.setProperty("--color-border-secondary", colors.border.secondary);
    root.style.setProperty("--color-border-light", colors.border.light);
    
    // Component colors
    root.style.setProperty("--color-card-bg", colors.card.bg);
    root.style.setProperty("--color-card-border", colors.card.border);
    root.style.setProperty("--color-navbar-bg", colors.navbar.bg);
    root.style.setProperty("--color-navbar-text", colors.navbar.text);
    root.style.setProperty("--color-input-bg", colors.input.bg);
    root.style.setProperty("--color-input-border", colors.input.border);
    root.style.setProperty("--color-input-text", colors.input.text);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const setThemeTo = (themeName) => {
    setTheme(themeName);
  };

  const values = {
    theme,
    toggleTheme,
    setThemeTo,
    isDark: theme === "dark",
  };

  return (
    <ThemeContext.Provider value={values}>
      {children}
    </ThemeContext.Provider>
  );
};
