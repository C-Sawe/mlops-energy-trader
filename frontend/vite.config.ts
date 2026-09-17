import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FR-18–FR-20's dashboard. /api is proxied to the FastAPI gateway
// (src/serving/api.py) so the dev server never needs CORS configured for
// itself — CORSMiddleware in api.py is for direct browser calls in a
// built/served deployment, not for this proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
