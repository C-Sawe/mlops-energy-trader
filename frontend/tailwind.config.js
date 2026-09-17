/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
      },
      colors: {
        // iOS's off-white system background, so a crisp white glass card
        // actually has something to stand out against.
        canvas: {
          light: "#F2F2F7",
          dark: "#000000",
        },
      },
      borderRadius: {
        "4xl": "28px",
      },
    },
  },
  plugins: [],
};
