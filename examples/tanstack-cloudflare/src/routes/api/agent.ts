// The substructure worker webhook. The ENGINE posts decision requests here
// (server-to-server); this returns the worker's actions. Point the engine at
// it: `substructure start --worker-url https://<app>/api/agent`.
import { createFileRoute } from "@tanstack/react-router";

import { handleAgentRequest } from "../../../substructure";

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => handleAgentRequest(request),
        },
    },
});
