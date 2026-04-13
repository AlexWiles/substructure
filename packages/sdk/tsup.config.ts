import { defineConfig } from "tsup";

export default defineConfig({
    entry: ["src/*.ts"],
    external: ["@substructure.ai/runtime"],
    format: ["esm"],
    dts: true,
    sourcemap: true,
    clean: true,
    outDir: "dist",
});
