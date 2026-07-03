// The Substructure worker webhook. The engine POSTs decision requests here
// (server-to-server); point it at this route with
// `substructure start --worker-url https://<app>/api/agent`.
import { createFileRoute } from "@tanstack/react-router";

import { substructureHandler } from "../../substructure";

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => substructureHandler(request),
        },
    },
});
