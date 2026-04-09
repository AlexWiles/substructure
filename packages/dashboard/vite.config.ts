import { defineConfig } from "vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import viteReact from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import tailwindcss from "@tailwindcss/vite";

const config = defineConfig({
    plugins: [tsconfigPaths({ projects: ["./tsconfig.json"] }), tailwindcss(), TanStackRouterVite(), viteReact()],
    server: {
        port: 3000,
        proxy: {
            "/admin": "http://localhost:8080",
        },
    },
});

export default config;
