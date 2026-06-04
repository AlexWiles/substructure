import { createFileRoute, redirect } from "@tanstack/react-router";

// The sidebar (in PageShell) is the nav now; open straight into the first
// client rather than a separate landing page.
export const Route = createFileRoute("/")({
    beforeLoad: () => {
        throw redirect({ to: "/assistant-ui" });
    },
});
