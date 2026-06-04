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

import { makeAssistantUiHistory } from "../lib/assistant-ui-history";
import { getColor, setColor, toHex } from "../lib/color-store";
import { useSessions } from "../lib/sessions";
import type { BrowserSession } from "../lib/token";
import { MarkdownText } from "./markdown-text";
import { ChatLoading } from "./page-shell";

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

type WeatherArgs = { city?: string };
type WeatherResult = { city: string; temp_f: number; condition: string };

// Card for the server-side get_weather tool. Without a registered UI a restored
// tool call falls back to assistant-ui's generic JSON card, so we give it one
// that matches the color swatches — same look live and rehydrated from history.
const WeatherToolUI = makeAssistantToolUI<WeatherArgs, WeatherResult>({
    toolName: "get_weather",
    render: ({ args, result, status }) => {
        const city = result?.city ?? args?.city;
        return (
            <div className="toolcard">
                <span className="toolcard-name">get_weather</span>
                {city ? <code className="toolcard-hex">{city}</code> : null}
                {result ? (
                    <span className="toolcard-status">
                        {result.temp_f}°F, {result.condition}
                    </span>
                ) : status.type !== "complete" ? (
                    <span className="toolcard-status">…</span>
                ) : null}
            </div>
        );
    },
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

// Mount the thread keyed on the active session so switching gives a fresh
// runtime. History is loaded by assistant-ui's history adapter (it ignores the
// agent's initialMessages); the agent's threadId targets the same session.
export function AssistantUiChat({ session }: { session: BrowserSession }) {
    const { activeId } = useSessions();
    if (!activeId) return <ChatLoading />;
    return <AssistantUiThread key={activeId} session={session} sessionId={activeId} />;
}

function AssistantUiThread({ session, sessionId }: { session: BrowserSession; sessionId: string }) {
    const { token, substructureUrl, agentId } = session;

    const agent = useMemo(
        () =>
            new HttpAgent({
                url: `${substructureUrl}/api/client/ag-ui/agents/${agentId}/run`,
                headers: { Authorization: `Bearer ${token}` },
                threadId: sessionId,
                // Bound fetch — HttpAgent calls `this.fetch(...)`, which throws
                // "Illegal invocation" in Firefox with an unbound reference.
                fetch: (url, init) => fetch(url, init),
            }),
        [token, substructureUrl, agentId, sessionId],
    );

    const adapters = useMemo(() => ({ history: makeAssistantUiHistory(session, sessionId) }), [session, sessionId]);

    const runtime = useAgUiRuntime({ agent, adapters });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <SetColor />
            <GetColor />
            <SetColorToolUI />
            <GetColorToolUI />
            <WeatherToolUI />
            <Thread assistantMessage={{ components: { Text: MarkdownText } }} />
        </AssistantRuntimeProvider>
    );
}
