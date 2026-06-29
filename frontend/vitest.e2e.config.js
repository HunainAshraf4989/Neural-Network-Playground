import { defineConfig } from "vitest/config";

// End-to-end suite: spawns the real standalone backend and drives the websocket
// protocol with the frontend's own message builders. Node environment (no DOM),
// generous timeouts because it boots a Python subprocess.
export default defineConfig({
  test: {
    environment: "node",
    globals: true,
    include: ["test/e2e/**/*.test.js"],
    testTimeout: 30000,
    hookTimeout: 30000,
    fileParallelism: false,
  },
});
