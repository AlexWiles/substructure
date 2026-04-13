import { defineConfig } from "tsup";

export default defineConfig({
    entry: [
        "src/client.ts",
        "src/admin-client.ts",
        "src/backend-client.ts",
        "src/frontend-client.ts",
        "src/worker-client.ts",
        "src/user-client.ts",
        "src/types.ts",
        "src/worker.ts",
        "src/runtime.ts",
        "src/run-stream.ts",
        "src/substructure.ts",
    ],
    external: ["@substructure.ai/runtime"],
    format: ["esm"],
    dts: true,
    clean: true,
    outDir: "dist",
});
