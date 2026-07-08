import { createFileRoute } from "@tanstack/react-router";

import { CopilotKitChat } from "../components/copilotkit-chat";
import { PageShell } from "../components/page-shell";
import { mintToken } from "../lib/token";

export const Route = createFileRoute("/copilotkit")({
    loader: () => mintToken(),
    component: Page,
});

function Page() {
    const session = Route.useLoaderData();
    return (
        <PageShell>
            <CopilotKitChat session={session} />
        </PageShell>
    );
}
