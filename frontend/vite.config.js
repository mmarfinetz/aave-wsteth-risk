import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  },
  server: {
    proxy: {
      '/api': {
        // Set VITE_API_PROXY=http://127.0.0.1:5001 to develop against a
        // local `python api.py`; default mirrors the deployed rewrite
        // target (see vercel.json).
        target: process.env.VITE_API_PROXY || 'https://aave-wsteth-risk-production.up.railway.app',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY || 'https://aave-wsteth-risk-production.up.railway.app',
        changeOrigin: true,
      },
    },
  },
})
