import { CopilotChat, CopilotKitProvider, HttpAgent } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { useMemo } from "react";

import { getColor, setColor, toHex } from "../lib/color-store";
import type { BrowserSession } from "../lib/token";

// Frontend tools. CopilotKit executes them after RUN_FINISHED off the standard
// TOOL_CALL_* events (no dialect hint needed). The schema is declared on the
// worker, so the handlers just read/write the shared color mixer.
const setColorTool = {
    name: "set_color",
    description: "Set the color in the on-screen color mixer (red/green/blue, 0–255).",
    handler: async (args: Record<string, unknown>) => {
        const { red, green, blue } = args as { red: number; green: number; blue: number };
        const c = setColor({ r: red, g: green, b: blue }, { animate: true });
        return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
    },
};

const getColorTool = {
    name: "get_color",
    description: "Read the color currently shown in the on-screen color mixer.",
    handler: async () => {
        const c = getColor();
        return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
    },
};

// CopilotKit v2 connects straight to an AG-UI endpoint via a self-managed
// HttpAgent — no CopilotRuntime broker. HttpAgent is re-exported from
// @copilotkit/react-core/v2 (its own @ag-ui/client version).
export function CopilotKitChat({ session }: { session: BrowserSession }) {
    const { token, substructureUrl, agentId } = session;

    const agent = useMemo(
        () => new HttpAgent({ url: `${substructureUrl}/api/client/ag-ui/agents/${agentId}/run` }),
        [substructureUrl, agentId],
    );

    // Auth goes on the provider, not the agent: CopilotKit overwrites a managed
    // agent's headers from the provider config.
    return (
        <CopilotKitProvider
            headers={{ Authorization: `Bearer ${token}` }}
            selfManagedAgents={{ [agentId]: agent }}
            frontendTools={[setColorTool, getColorTool]}
        >
            <CopilotChat agentId={agentId} />
        </CopilotKitProvider>
    );
}
