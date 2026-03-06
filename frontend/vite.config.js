import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // все запросы с фронта на /analyze и /documents будут проксированы на Go backend
      '/analyze': 'http://backend:8080',
      '/documents': 'http://backend:8080',
    },
  },
})
