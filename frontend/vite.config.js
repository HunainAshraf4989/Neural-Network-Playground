import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend's websocket port lives in exactly one place: DEFAULT_WS_PORT in
// backend/config.py. Read it from there at dev/build time so changing that one
// number moves the canvas with it, instead of leaving the UI knocking on a port
// nobody is serving. Falls back to 8765 if the file isn't readable (e.g. the
// frontend checked out on its own).
function backendWsPort() {
  const configPath = fileURLToPath(new URL("../backend/config.py", import.meta.url));
  try {
    const match = readFileSync(configPath, "utf8").match(/^DEFAULT_WS_PORT\s*=\s*(\d+)/m);
    if (match) return Number(match[1]);
  } catch {
    // unreadable - fall through to the default
  }
  return 8765;
}

// Unit/component tests run in jsdom. The live-backend e2e suite lives under
// test/e2e and is excluded here; run it with `npm run test:e2e`.
export default defineConfig({
  plugins: [react()],
  define: { __NN_WS_PORT__: JSON.stringify(backendWsPort()) },
  server: { port: 5173 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./test/setup.js"],
    include: ["src/**/*.test.{js,jsx}"],
  },
});
