import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The frontend lives in web/ and builds to web/dist, which server.ts serves.
export default defineConfig({
    root: "web",
    build: { outDir: "dist", emptyOutDir: true },
    plugins: [react()],
});
