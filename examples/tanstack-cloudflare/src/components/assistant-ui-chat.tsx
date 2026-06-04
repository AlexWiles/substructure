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

// Render a color tool call inside the thread as a swatch card. assistant-ui shows
// tool calls via a renderer registered with makeAssistantToolUI (matched by
// toolName); mounting the returned component registers it. Without one, <Thread />
// falls back to a generic tool box. The same card serves set_color and get_color
// — it reads the resolved color from `result` (or the streaming `args`).
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

// NOTE: assistant-ui resumes a frontend tool only when no run is in flight
// (`addToolResult` → `if (!isRunningFlag) startRun`). A fast client tool can
// resolve before the originating run's SSE stream finishes closing, so the
// resume gets dropped and the chat hangs after the tool runs. This is a client
// bug (the endpoint emits the spec-standard stream); tracked upstream in
// @assistant-ui/react-ag-ui. The small delay below keeps the demo working
// against it by letting the run close first.
//
// Frontend tool. assistant-ui executes it off the standard TOOL_CALL_* events
// (no dialect hint needed — the endpoint's default spec output is enough). It
// drives the shared color mixer.
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
        // See the note above re the upstream resume race.
        await new Promise((r) => setTimeout(r, 400));
        const c = setColor({ r: red, g: green, b: blue }, { animate: true });
        return { red: c.r, green: c.g, blue: c.b, hex: toHex(c) };
    },
});

// The read counterpart: returns whatever the user currently has in the mixer.
const GetColor = makeAssistantTool({
    toolName: "get_color",
    type: "frontend",
    description: "Read the color currently shown in the on-screen color mixer.",
    parameters: { type: "object", properties: {} },
    execute: async () => {
        // See the note above re the upstream resume race.
        await new Promise((r) => setTimeout(r, 400));
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
                // HttpAgent calls fetch as `this.fetch(...)`; hand it a bound one
                // so Firefox doesn't throw "Illegal invocation".
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
