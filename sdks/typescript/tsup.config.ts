import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/client.ts",
    "src/admin-client.ts",
    "src/worker-client.ts",
    "src/user-client.ts",
    "src/types.ts",
    "src/agent.ts",
    "src/worker.ts",
  ],
  format: ["esm"],
  dts: true,
  clean: true,
  outDir: "dist",
});
