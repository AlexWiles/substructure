import { cloudflare } from "@cloudflare/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

// `pnpm build:cf` / `pnpm deploy` bundle for the Workers runtime; plain `vite` runs the Node target.
const isCloudflareBuild = process.env.DEPLOY_TARGET === "cloudflare";

export default defineConfig({
    // strictPort so a busy port fails loudly instead of silently moving; 3030 dodges common 3000/3001 collisions.
    server: { port: 3030, strictPort: true },
    plugins: [
        ...(isCloudflareBuild ? [cloudflare({ viteEnvironment: { name: "ssr" } })] : []),
        tsconfigPaths(),
        tailwindcss(),
        tanstackStart(),
        viteReact(),
    ],
});
