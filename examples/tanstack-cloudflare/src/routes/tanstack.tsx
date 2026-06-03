import { createFileRoute } from "@tanstack/react-router";

import { PageShell } from "../components/page-shell";
import { TanStackChat } from "../components/tanstack-chat";
import { mintToken } from "../lib/token";

export const Route = createFileRoute("/tanstack")({
    loader: () => mintToken(),
    component: Page,
});

function Page() {
    const session = Route.useLoaderData();
    return (
        <PageShell>
            <TanStackChat session={session} />
        </PageShell>
    );
}
