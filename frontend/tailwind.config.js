export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#f0f0f0",
          100: "#e0e0e0",
          200: "#c0c0c0",
          300: "#a0a0a0",
          400: "#808080",
          500: "#1C9BC9",
          600: "#1875ad",
          700: "#0e4a66",
          800: "#083140",
          900: "#030a0f",
        },
      },
      keyframes: {
        fadeInUp: {
          from: {
            opacity: "0",
            transform: "translateY(10px)",
          },
          to: {
            opacity: "1",
            transform: "translateY(0)",
          },
        },
        fadeIn: {
          from: {
            opacity: "0",
          },
          to: {
            opacity: "1",
          },
        },
      },
      animation: {
        fadeInUp: "fadeInUp 0.3s ease-out",
        fadeIn: "fadeIn 0.2s ease-out",
      },
      transitionDuration: {
        250: "250ms",
      },
    },
  },
  plugins: [],
}