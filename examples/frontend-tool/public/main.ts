// Runs in the browser. Bundled to public/main.js by esbuild.
//
// Flow per turn:
//   1. POST /token  -> ask the backend to mint a short-lived JWT.
//   2. sub.frontend.client({ token }).startTurn(...)
//   3. for-await over client.stream(scope):
//        - on `tool.call.requested` for a tool we know how to run, execute
//          it in the browser and POST the result back via
//          submitToolCallResult — the agent resumes as if the tool had
//          returned synchronously from the worker.
//        - on `message.new` (assistant), render the message.
//
// The worker also gets the tool.execute trigger; its `execute` returns
// ctx.defer() so no result action is emitted server-side and the engine
// waits for the browser to deliver one.

import Substructure, { isTokenDelta } from "@substructure.ai/sdk";

const chatLog = document.getElementById("chat") as HTMLDivElement;
const form = document.getElementById("form") as HTMLFormElement;
const input = document.getElementById("input") as HTMLInputElement;
const sendBtn = document.getElementById("send") as HTMLButtonElement;

type FrontendTool = (args: any) => Promise<unknown> | unknown;

const tools: Record<string, FrontendTool> = {
    get_user_location: () =>
        new Promise((resolve) => {
            if (!navigator.geolocation) {
                resolve({ error: "Geolocation not supported in this browser." });
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (pos) =>
                    resolve({
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy_m: pos.coords.accuracy,
                    }),
                (err) => resolve({ error: err.message }),
                { timeout: 10_000 },
            );
        }),
    set_theme: ({ background, accent }: { background: string; accent: string }) => {
        const root = document.documentElement.style;
        if (background) root.setProperty("--bg", background);
        if (accent) root.setProperty("--accent", accent);
        return { ok: true, background, accent };
    },
};

function append(role: "user" | "assistant" | "tool", text: string) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
}

function showTyping() {
    const div = document.createElement("div");
    div.className = "msg assistant typing";
    div.innerHTML = "<span></span><span></span><span></span>";
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
}

async function fetchToken() {
    const r = await fetch("/token", { method: "POST" });
    if (!r.ok) throw new Error(`/token failed: ${r.status} ${await r.text()}`);
    return (await r.json()) as { token: string; expiresAt: number; substructureUrl: string };
}

const sub = new Substructure();

let client: ReturnType<typeof sub.frontend.client>;
try {
    const { token, substructureUrl } = await fetchToken();
    client = sub.frontend.client({ token, url: substructureUrl });
} catch (err: any) {
    append(
        "tool",
        `✗ couldn't mint a client token: ${err?.message ?? err}. is \`substructure start --dev --port 9000 --worker-url http://localhost:3333/agent\` running?`,
    );
    throw err;
}

let sessionId: string | undefined;

async function sendMessage(content: string) {
    append("user", content);
    sendBtn.disabled = true;
    input.disabled = true;

    let typing: HTMLDivElement | null = showTyping();
    // Per-call partial bubble. Chunks may arrive out of order, so buffer
    // by seq and flush contiguous prefix.
    type Partial = { node: HTMLDivElement; chunks: Map<number, string>; nextSeq: number };
    const partials = new Map<string, Partial>();

    const flush = (p: Partial) => {
        let chunk = p.chunks.get(p.nextSeq);
        while (chunk !== undefined) {
            p.node.textContent = (p.node.textContent ?? "") + chunk;
            p.chunks.delete(p.nextSeq);
            p.nextSeq += 1;
            chunk = p.chunks.get(p.nextSeq);
        }
        chatLog.scrollTop = chatLog.scrollHeight;
    };

    try {
        const scope = await client.startTurn({
            agentId: "browser-assistant",
            payload: { type: "message", message: { role: "user", content } },
            sessionId,
        });
        sessionId = scope.sessionId;

        for await (const event of client.stream(scope, { tokens: true })) {
            if (isTokenDelta(event)) {
                let partial = partials.get(event.call_id);
                if (!partial) {
                    typing?.remove();
                    typing = null;
                    partial = { node: append("assistant", ""), chunks: new Map(), nextSeq: 0 };
                    partials.set(event.call_id, partial);
                }
                if (typeof event.text === "string" && event.text.length > 0) {
                    partial.chunks.set(event.seq, event.text);
                    flush(partial);
                }
                continue;
            }

            if (event.payload.type === "llm.call.completed") {
                const partial = partials.get(event.payload.call_id);
                if (partial) {
                    partial.node.remove();
                    partials.delete(event.payload.call_id);
                }
            } else if (event.payload.type === "tool.call.requested" && tools[event.payload.name]) {
                const { tool_call_id, name, arguments: argsJson, attempt } = event.payload;
                const args = argsJson ? JSON.parse(argsJson) : {};
                typing?.remove();
                typing = null;
                append("tool", `→ ${name}(${argsJson || ""})`);
                try {
                    const result = await tools[name](args);
                    append("tool", `← ${JSON.stringify(result)}`);
                    await client.submitToolCallResult({
                        sessionId: scope.sessionId,
                        toolCallId: tool_call_id,
                        attempt,
                        result: JSON.stringify(result),
                    });
                } catch (err: any) {
                    const message = err?.message ?? String(err);
                    append("tool", `✗ ${message}`);
                    await client.submitToolCallResult({
                        sessionId: scope.sessionId,
                        toolCallId: tool_call_id,
                        attempt,
                        error: message,
                        retryable: false,
                    });
                }
                typing = showTyping();
            } else if (event.payload.type === "message.new" && event.payload.message.role === "assistant") {
                const text = typeof event.payload.message.content === "string" ? event.payload.message.content : "";
                if (text.trim()) {
                    typing?.remove();
                    typing = null;
                    append("assistant", text);
                }
            }
        }
    } catch (err: any) {
        append("tool", `✗ ${err?.message ?? String(err)}`);
    } finally {
        typing?.remove();
        sendBtn.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    void sendMessage(text);
});

input.placeholder = "ask me anything…";
input.disabled = false;
sendBtn.disabled = false;
input.focus();
