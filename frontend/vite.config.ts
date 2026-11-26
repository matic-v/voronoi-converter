import { defineConfig } from "vite";

// In dev mode (npm run dev) API calls are proxied to the Python server.
export default defineConfig({
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
