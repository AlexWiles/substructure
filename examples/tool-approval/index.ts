// Tool-call approval gate over real shell access: when the model calls
// `run_command` the handler parks it and ends the turn; the client approves or
// denies, and on approval the call is re-emitted so the model sees one matched pair.
//
// WARNING: this example runs real shell commands. Approve carefully.

import { spawnSync } from "node:child_process";
import { agent, type Message, tool, toolLoop } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

// ── State (rides the wire) ───────────────────────────────────────────────────

type State = {
    pendingCommand: { toolCallId: string; cmd: string } | null;
    approvalDecision: { approved: boolean; reason?: string } | null;
};

const instructions =
    "You are a shell assistant. Use `run_command` to run real shell commands on the user's machine. " +
    "Every command requires explicit user approval before it runs and may be denied with a reason. " +
    "If a command is denied, adapt rather than retrying the same command.";

function runShell(cmd: string): string {
    const result = spawnSync(cmd, { shell: true, encoding: "utf8", timeout: 30_000, maxBuffer: 1_000_000 });
    return JSON.stringify({
        cmd,
        exit_code: result.status ?? -1,
        stdout: result.stdout ?? "",
        stderr: result.stderr ?? result.error?.message ?? "",
    });
}

// Reads the decision the gate recorded: run the command, or return the denial so the model can adapt.
function commandTool(state: State) {
    return tool({
        name: "run_command",
        description:
            "Run a shell command on the host machine. Requires user approval — the call pauses until the user approves or denies. Output is the real exit code, stdout, and stderr.",
        parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] },
        execute: (args) => {
            const { cmd } = JSON.parse(args) as { cmd: string };
            const decision = state.approvalDecision;
            state.approvalDecision = null;
            if (decision && !decision.approved) {
                return JSON.stringify({
                    cmd,
                    exit_code: 1,
                    stdout: "",
                    stderr: `User denied this command.${decision.reason ? ` Reason: ${decision.reason}` : ""}`,
                });
            }
            return runShell(cmd);
        },
    });
}

// ── Agent ───────────────────────────────────────────────────────────────────
// The default loop drives the conversation; the handler overrides only the two gate triggers.

const assistant = agent<State>({
    name: "assistant",
    decide: async (req) => {
        const state: State = {
            pendingCommand: req.state?.pendingCommand ?? null,
            approvalDecision: req.state?.approvalDecision ?? null,
        };
        const loop = toolLoop<State>({
            llm: { model: "anthropic/claude-sonnet-4-6" },
            instructions,
            tools: [commandTool(state)],
        });

        // The model asked to run a command: record the turn, park the call, end it.
        if (req.trigger.type === "llm.finished" && req.trigger.message) {
            const message = req.trigger.message;
            const call = (message.tool_calls ?? []).find((tc) => tc.function.name === "run_command");
            if (call) {
                const assistantMsg: Message = { ...message, id: crypto.randomUUID() };
                state.pendingCommand = { toolCallId: call.id, cmd: JSON.parse(call.function.arguments).cmd };
                return {
                    transcript: [...(req.transcript ?? []), assistantMsg],
                    actions: [{ type: "done", data: { pendingCommand: state.pendingCommand } }],
                    state,
                };
            }
        }

        // The user approved or denied: re-emit the parked call so the loop runs it.
        if (req.trigger.type === "client.action" && req.trigger.name === "approve_command") {
            const pending = state.pendingCommand;
            if (!pending) return { actions: [{ type: "done", data: "no pending command to approve" }], state };
            const args = (req.trigger.args ?? {}) as { approved: boolean; reason?: string };
            state.approvalDecision = { approved: args.approved, reason: args.reason };
            state.pendingCommand = null;
            return {
                actions: [
                    {
                        type: "tool.call",
                        id: pending.toolCallId,
                        name: "run_command",
                        arguments: JSON.stringify({ cmd: pending.cmd }),
                        handler: "worker",
                    },
                ],
                state,
            };
        }

        // Prompting, running the approved command, and continuing are the default loop.
        // The tools mutate `state` while the loop runs, so return it to persist.
        const d = await loop({ ...req, state });
        return { ...d, state };
    },
});

// ── CLI driver ──────────────────────────────────────────────────────────────
// Usage:
//   pnpm tsx index.ts <session-id> "<message>"
//   pnpm tsx index.ts <session-id> /approve
//   pnpm tsx index.ts <session-id> /deny [reason]
// Reuse one session id across calls; sessions persist in agent.db.

const [, , sessionId, ...rest] = process.argv;
const input = rest.join(" ");
if (!sessionId || !input) {
    console.error('Usage: pnpm tsx index.ts <session-id> "<message>" | /approve | /deny [reason]');
    process.exit(1);
}

const payload =
    input === "/approve"
        ? { type: "action" as const, name: "approve_command", args: { approved: true } }
        : input === "/deny" || input.startsWith("/deny ")
          ? {
                type: "action" as const,
                name: "approve_command",
                args: { approved: false, reason: input.slice(5).trim() || undefined },
            }
          : { type: "message" as const, message: { role: "user" as const, content: input } };

const embedded = await SubstructureEmbedded.create({
    agents: [assistant],
    db: "agent.db",
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
});

const scope = await embedded.startTurn({
    agentId: "assistant",
    payload,
    identity: { tenant_id: "default", id: "demo" },
    sessionId,
});

const label =
    payload.type === "action"
        ? `action:${payload.name}(${JSON.stringify(payload.args)})`
        : `message:"${payload.message.content}"`;
console.log(`── ${label}\nsession ${scope.sessionId}\n`);

for await (const event of embedded.stream(scope)) {
    const p = event.payload;
    switch (p.type) {
        case "llm.call.requested":
            console.log(`  → llm.call.requested (${p.request.model})`);
            break;
        case "llm.call.completed": {
            const text = p.response.content;
            if (text) console.log(`  ← llm: ${text.trim().replace(/\n+/g, " ⏎ ")}`);
            break;
        }
        case "tool.call.requested":
            console.log(`  → tool.call ${p.name}(${p.arguments})`);
            break;
        case "tool.call.completed":
            console.log(`  ← tool.result ${p.name}: ${p.result}`);
            break;
        case "tool.call.errored":
            console.log(`  ✗ tool.error ${p.name}: ${p.error}`);
            break;
        case "turn.completed": {
            const dataObj = (p.data ?? {}) as { pendingCommand?: { cmd: string } };
            if (dataObj.pendingCommand) {
                console.log(`⏸ awaiting approval for: ${dataObj.pendingCommand.cmd}`);
                console.log(`   /approve  or  /deny [reason]  to continue`);
            }
            console.log(
                `✓ turn.completed${p.turn_cost ? `  cost=${p.turn_cost}` : ""}${
                    p.turn_token_usage ? `  tokens=${JSON.stringify(p.turn_token_usage)}` : ""
                }`,
            );
            break;
        }
    }
}

await embedded.shutdown();
