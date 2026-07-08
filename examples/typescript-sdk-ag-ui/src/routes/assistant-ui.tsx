import { createFileRoute } from "@tanstack/react-router";

import { AssistantUiChat } from "../components/assistant-ui-chat";
import { PageShell } from "../components/page-shell";
import { mintToken } from "../lib/token";

export const Route = createFileRoute("/assistant-ui")({
    loader: () => mintToken(),
    component: Page,
});

function Page() {
    const session = Route.useLoaderData();
    return (
        <PageShell>
            <AssistantUiChat session={session} />
        </PageShell>
    );
}
