import { cloudflare } from "@cloudflare/vite-plugin";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

// `pnpm build:cf` / `pnpm deploy` set this to bundle for the Workers runtime.
// Plain `vite dev` / `vite build` run the Node target for local development.
const isCloudflareBuild = process.env.DEPLOY_TARGET === "cloudflare";

export default defineConfig({
    // strictPort so a busy port fails loudly instead of silently moving — then
    // the browser would hit whatever else owns the port (and 404). 3030 is
    // chosen to dodge the common 3000/3001 dev collisions.
    server: { port: 3030, strictPort: true },
    plugins: [
        ...(isCloudflareBuild ? [cloudflare({ viteEnvironment: { name: "ssr" } })] : []),
        tsconfigPaths(),
        tanstackStart(),
        viteReact(),
    ],
});
