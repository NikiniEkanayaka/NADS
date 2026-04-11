// Theme color definitions
export const themes = {
  light: {
    name: "Light",
    colors: {
      primary: "#1C9BC9",
      primaryDark: "#1583ad",
      secondary: "#6366f1",
      success: "#10b981",
      warning: "#f59e0b",
      danger: "#ef4444",
      info: "#3b82f6",
      
      // Background colors
      bg: {
        primary: "#ffffff",
        secondary: "#f9fafb",
        tertiary: "#f3f4f6",
        hover: "#f5f5f5",
      },
      
      // Text colors
      text: {
        primary: "#111827",
        secondary: "#6b7280",
        tertiary: "#9ca3af",
        inverse: "#ffffff",
      },
      
      // Border colors
      border: {
        primary: "#e5e7eb",
        secondary: "#d1d5db",
        light: "#f3f4f6",
      },
      
      // Component specific
      card: {
        bg: "#ffffff",
        border: "#e5e7eb",
      },
      navbar: {
        bg: "#ffffff",
        text: "#1f2937",
      },
      input: {
        bg: "#ffffff",
        border: "#d1d5db",
        text: "#111827",
      },
    },
  },
  
  dark: {
    name: "Dark",
    colors: {
      primary: "#0ea5e9",
      primaryDark: "#0284c7",
      secondary: "#818cf8",
      success: "#34d399",
      warning: "#fbbf24",
      danger: "#f87171",
      info: "#60a5fa",
      
      // Background colors
      bg: {
        primary: "#0f172a",
        secondary: "#1e293b",
        tertiary: "#334155",
        hover: "#1e293b",
      },
      
      // Text colors
      text: {
        primary: "#f1f5f9",
        secondary: "#cbd5e1",
        tertiary: "#94a3b8",
        inverse: "#0f172a",
      },
      
      // Border colors
      border: {
        primary: "#334155",
        secondary: "#475569",
        light: "#1e293b",
      },
      
      // Component specific
      card: {
        bg: "#1e293b",
        border: "#334155",
      },
      navbar: {
        bg: "#1e293b",
        text: "#f1f5f9",
      },
      input: {
        bg: "#0f172a",
        border: "#334155",
        text: "#f1f5f9",
      },
    },
  },

  ocean: {
    name: "Ocean",
    colors: {
      primary: "#0369a1",
      primaryDark: "#0c4a6e",
      secondary: "#0891b2",
      success: "#06b6d4",
      warning: "#ea580c",
      danger: "#dc2626",
      info: "#0ea5e9",
      
      bg: {
        primary: "#f0f9ff",
        secondary: "#e0f2fe",
        tertiary: "#cffafe",
        hover: "#e0f2fe",
      },
      
      text: {
        primary: "#082f49",
        secondary: "#164e63",
        tertiary: "#06b6d4",
        inverse: "#f0f9ff",
      },
      
      border: {
        primary: "#bae6fd",
        secondary: "#7dd3fc",
        light: "#e0f2fe",
      },
      
      card: {
        bg: "#ffffff",
        border: "#bae6fd",
      },
      navbar: {
        bg: "#ffffff",
        text: "#082f49",
      },
      input: {
        bg: "#f0f9ff",
        border: "#bae6fd",
        text: "#082f49",
      },
    },
  },

  forest: {
    name: "Forest",
    colors: {
      primary: "#15803d",
      primaryDark: "#166534",
      secondary: "#16a34a",
      success: "#22c55e",
      warning: "#84cc16",
      danger: "#dc2626",
      info: "#14b8a6",
      
      bg: {
        primary: "#f0fdf4",
        secondary: "#dcfce7",
        tertiary: "#bbf7d0",
        hover: "#dcfce7",
      },
      
      text: {
        primary: "#14532d",
        secondary: "#166534",
        tertiary: "#4d7c0f",
        inverse: "#f0fdf4",
      },
      
      border: {
        primary: "#86efac",
        secondary: "#4ade80",
        light: "#dcfce7",
      },
      
      card: {
        bg: "#ffffff",
        border: "#86efac",
      },
      navbar: {
        bg: "#ffffff",
        text: "#14532d",
      },
      input: {
        bg: "#f0fdf4",
        border: "#86efac",
        text: "#14532d",
      },
    },
  },

  sunset: {
    name: "Sunset",
    colors: {
      primary: "#ea580c",
      primaryDark: "#c2410c",
      secondary: "#f97316",
      success: "#10b981",
      warning: "#f59e0b",
      danger: "#ef4444",
      info: "#f43f5e",
      
      bg: {
        primary: "#fefce8",
        secondary: "#fef08a",
        tertiary: "#fde047",
        hover: "#fef08a",
      },
      
      text: {
        primary: "#713f12",
        secondary: "#92400e",
        tertiary: "#b45309",
        inverse: "#fefce8",
      },
      
      border: {
        primary: "#fcd34d",
        secondary: "#fbbf24",
        light: "#fef08a",
      },
      
      card: {
        bg: "#ffffff",
        border: "#fcd34d",
      },
      navbar: {
        bg: "#ffffff",
        text: "#713f12",
      },
      input: {
        bg: "#fefce8",
        border: "#fcd34d",
        text: "#713f12",
      },
    },
  },
};

export const getTheme = (themeName) => {
  return themes[themeName] || themes.light;
};

export const themeList = Object.entries(themes).map(([key, value]) => ({
  id: key,
  name: value.name,
}));
