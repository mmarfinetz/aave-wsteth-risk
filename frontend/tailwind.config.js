/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          bg: '#0a0b0d',
          card: '#111318',
          hover: '#181b22',
        },
        accent: {
          gold: '#f0b429',
          teal: '#2dd4bf',
          coral: '#ef4444',
          orange: '#f97316',
        },
        txt: {
          primary: '#e8e6e3',
          secondary: '#6b7280',
          muted: '#4b5563',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        sans: ['"IBM Plex Sans"', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
