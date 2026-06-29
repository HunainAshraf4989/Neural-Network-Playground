import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Unit/component tests run in jsdom. The live-backend e2e suite lives under
// test/e2e and is excluded here; run it with `npm run test:e2e`.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./test/setup.js"],
    include: ["src/**/*.test.{js,jsx}"],
  },
});
