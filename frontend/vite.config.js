import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // все запросы с фронта на /analyze будут проксированы на Go backend
      '/analyze': 'http://expert-backend:8080',
    },
  },
})
