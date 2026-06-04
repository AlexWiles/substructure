import { HttpAgent } from "@ag-ui/client";
import {
    AssistantRuntimeProvider,
    makeAssistantTool,
    makeAssistantToolUI,
    type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { Thread } from "@assistant-ui/react-ui";
import "@assistant-ui/react-ui/styles/index.css";
import { useMemo } from "react";

import { getColor, setColor, toHex } from "../lib/color-store";
import type { BrowserSession } from "../lib/token";

type ColorArgs = { red?: number; green?: number; blue?: number };
type ColorResult = { red: number; green: number; blue: number; hex: string };

// Swatch card for a color tool call, shared by set_color and get_color. Reads
// the resolved color from `result`, falling back to the streaming `args`.
function colorCard({ toolName, args, result, status }: ToolCallMessagePartProps<ColorArgs, ColorResult>) {
    const r = result?.red ?? args?.red;
    const g = result?.green ?? args?.green;
    const b = result?.blue ?? args?.blue;
    const known = [r, g, b].every((n) => typeof n === "number");
    const hex = result?.hex ?? (known ? toHex({ r: r!, g: g!, b: b! }) : undefined);
    return (
        <div className="toolcard">
            <span className="toolcard-swatch" style={{ background: hex ?? "#ddd" }} />
            <span className="toolcard-name">{toolName}</span>
            {hex ? <code className="toolcard-hex">{hex.toUpperCase()}</code> : null}
            {status.type !== "complete" ? <span className="toolcard-status">…</span> : null}
        </div>
    );
}

const SetColorToolUI = makeAssistantToolUI<ColorArgs, ColorResult>({
    toolName: "set_color",
    render: colorCard,
});

const GetColorToolUI = makeAssistantToolUI<ColorArgs, ColorResult>({
    toolName: "get_color",
    render: colorCard,
});

// Frontend tool driving the shared color mixer. The execute() delay works
// around an upstream race: assistant-ui resumes a frontend tool only when no run
// is in flight, but a fast tool can resolve before the originating run's SSE
// stream closes, dropping the resume and hanging the chat. The delay lets the
// run close first. Tracked upstream in @assistant-ui/react-ag-ui.
const SetColor = makeAssistantTool({
    toolName: "set_color",
    type: "frontend",
    description: "Set the color in the on-screen color mixer (red/green/blue, 0–255).",
    parameters: {
        type: "object",
        properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" },
        },
        required: ["red", "green", "blue"],
    },
    execute: async (args) => {
        const { red, green, blue } = args as { red: number; green: number; blue: number };
        await new Promise((r) => setTimeout(r, 400)); // see SetColor note re resume race

        const c = setColor({ r: red, g: green, b: blue }, { animate: true });
        return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
    },
});

const GetColor = makeAssistantTool({
    toolName: "get_color",
    type: "frontend",
    description: "Read the color currently shown in the on-screen color mixer.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
        await new Promise((r) => setTimeout(r, 400)); // see SetColor note re resume race
        const c = getColor();
        return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
    },
});

export function AssistantUiChat({ session }: { session: BrowserSession }) {
    const { token, substructureUrl, agentId } = session;

    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${substructureUrl}/api/client/ag-ui/agents/${agentId}/run`,
                headers: { Authorization: `Bearer ${token}` },
                // Bound fetch — HttpAgent calls `this.fetch(...)`, which throws
                // "Illegal invocation" in Firefox with an unbound reference.
                fetch: (url, init) => fetch(url, init),
            }),
        [token, substructureUrl, agentId],
    );

    const runtime = useAgUiRuntime({ agent });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <SetColor />
            <GetColor />
            <SetColorToolUI />
            <GetColorToolUI />
            <Thread />
        </AssistantRuntimeProvider>
    );
}
