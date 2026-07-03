import { HeadContent, Outlet, Scripts, createRootRoute } from "@tanstack/react-router";
import type { ReactNode } from "react";

import "../styles.css";

export const Route = createRootRoute({
    head: () => ({
        meta: [
            { charSet: "utf-8" },
            { name: "viewport", content: "width=device-width, initial-scale=1" },
            { title: "Substructure × assistant-ui" },
        ],
    }),
    component: () => <Outlet />,
    shellComponent: RootDocument,
});

function RootDocument({ children }: { children: ReactNode }) {
    return (
        <html lang="en" className="h-dvh">
            <head>
                <HeadContent />
            </head>
            <body className="h-dvh font-sans">
                {children}
                <Scripts />
            </body>
        </html>
    );
}
