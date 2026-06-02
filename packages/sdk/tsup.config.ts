import { readFileSync } from "node:fs";
import { defineConfig } from "tsup";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")) as {
    version: string;
};

export default defineConfig({
    entry: ["src/*.ts"],
    external: ["@substructure.ai/runtime"],
    format: ["esm"],
    // Declarations are emitted by `tsc` (see the build script) so we get
    // per-file `.d.ts` + `.d.ts.map` declaration maps. tsup's bundled dts
    // can't produce declaration maps, which breaks go-to-source for consumers.
    dts: false,
    sourcemap: true,
    clean: true,
    outDir: "dist",
    define: {
        __SDK_VERSION__: JSON.stringify(pkg.version),
    },
});
