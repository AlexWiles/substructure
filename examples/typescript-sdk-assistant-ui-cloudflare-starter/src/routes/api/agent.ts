// The Substructure worker webhook: the engine POSTs decision requests here.
// Point the engine at this route: `substructure start --worker-url https://<app>/api/agent`.
import { createFileRoute } from "@tanstack/react-router";

import { substructureHandler } from "../../substructure";

export const Route = createFileRoute("/api/agent")({
    server: {
        handlers: {
            POST: ({ request }) => substructureHandler(request),
        },
    },
});
