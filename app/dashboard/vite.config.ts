import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the twin backend (app/server.py, default port
// 8734). In production the FastAPI server serves dist/ itself, so the
// client always talks to same-origin relative paths.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8734",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8734",
        ws: true,
      },
    },
  },
});
