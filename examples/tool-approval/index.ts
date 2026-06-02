// Tool-call approval against real shell access: the agent has
// `run_command`, a tool that actually executes commands on the host
// via `child_process.spawnSync`. Before any command runs, the
// `approvalGate` middleware intercepts the `call.tool`, parks it in
// state, and ends the turn. The client decides to approve or deny via
// a `client.action approve_command` payload; on resume the middleware
// re-emits the original `call.tool` so the LLM sees a single matched
// pair (to the model it just looks like a slow tool call).
//
// Denials carry an optional reason that surfaces to the LLM as part of
// the tool result, so it can adapt (try a different command, ask the
// user, or give up on the task).
//
// WARNING: this example runs real shell commands. Approve carefully.

import { spawnSync } from "node:child_process";
import Substructure, { DEFAULT_RETRY, middleware } from "@substructure.ai/sdk";
import { SubstructureEmbedded } from "@substructure.ai/sdk/embedded";

const sub = new Substructure();
const { agent } = sub;

// ── State ───────────────────────────────────────────────────────────────────

type ApprovalState = {
    pendingCommand: { toolCallId: string; cmd: string } | null;
    approvalDecision: { approved: boolean; reason?: string } | null;
};

const initialState: ApprovalState = {
    pendingCommand: null,
    approvalDecision: null,
};

// ── approvalGate middleware ─────────────────────────────────────────────────
// Owns the shared state slice (`{ pendingCommand, approvalDecision }`) and
// the gating logic: sits OUTER of `agent.tools` so it sees the `call.tool`
// actions the tools middleware emits. On the way out, if any action is
// `call.tool` for `run_command`, park it in state and end the turn.

const approvalGate = middleware<ApprovalState>({
    state: initialState,
    handler: async (ctx, next) => {
        const result = await next(ctx);

        let intercepted = false;
        const kept = result.actions.filter((action) => {
            if (action.type === "call.tool" && action.name === "run_command") {
                const a = JSON.parse(action.arguments) as { cmd: string };
                ctx.state.pendingCommand = { toolCallId: action.tool_call_id, cmd: a.cmd };
                intercepted = true;
                return false;
            }
            return true;
        });

        if (!intercepted) return result;

        // Drop any pending `call.llm` so the turn truly pauses, then end it
        // with a `done` action carrying the pending command for the client.
        const finalActions = kept.filter((a) => a.type !== "call.llm");
        finalActions.push({
            type: "done",
            data: { pendingCommand: ctx.state.pendingCommand },
        });
        return { ...result, actions: finalActions };
    },
});

// ── Approval action ─────────────────────────────────────────────────────────
// On `client.action approve_command`: write the decision, re-emit the
// original `call.tool` so it flows through the normal tool execution path
// (and `run_command` reads the decision to either run or return a denial).

const approveCommand = agent.action({
    name: "approve_command",
    state: approvalGate,
    handler: (args: { approved: boolean; reason?: string }, state) => {
        const pending = state.pendingCommand;
        if (!pending) return [{ type: "done", data: "no pending command to approve" }];

        state.approvalDecision = { approved: args.approved, reason: args.reason };
        state.pendingCommand = null;
        return [
            {
                type: "call.tool",
                tool_call_id: pending.toolCallId,
                name: "run_command",
                arguments: JSON.stringify({ cmd: pending.cmd }),
                handler: "worker",
                retry: DEFAULT_RETRY,
            },
        ];
    },
});

// ── Tool ────────────────────────────────────────────────────────────────────

const runCommand = agent.tool({
    name: "run_command",
    description:
        "Run a shell command on the host machine. Requires user approval — the call pauses until the user approves or denies. Output is the real exit code, stdout, and stderr.",
    parameters: {
        type: "object",
        properties: { cmd: { type: "string" } },
        required: ["cmd"],
    },
    state: approvalGate,
    execute: (args, state) => {
        const { cmd } = JSON.parse(args) as { cmd: string };
        const decision = state.approvalDecision;
        state.approvalDecision = null;
        if (decision && !decision.approved) {
            const reason = decision.reason ? ` Reason: ${decision.reason}` : "";
            return {
                cmd,
                exit_code: 1,
                stdout: "",
                stderr: `User denied this command.${reason}`,
            };
        }
        const result = spawnSync(cmd, {
            shell: true,
            encoding: "utf8",
            timeout: 30_000,
            maxBuffer: 1_000_000,
        });
        return {
            cmd,
            exit_code: result.status ?? -1,
            stdout: result.stdout ?? "",
            stderr: result.stderr ?? result.error?.message ?? "",
        };
    },
});

// ── Agent ───────────────────────────────────────────────────────────────────

const assistant = agent({ id: "assistant" })
    .use(agent.jsonState())
    .use(
        agent.systemMessage(
            "You are a shell assistant. Use `run_command` to run real shell commands on the user's machine. Every command requires explicit user approval before it runs and may be denied with a reason. If a command is denied, adapt rather than retrying the same command.",
        ),
    )
    .use(agent.messageHistory())
    .use(agent.actions([approveCommand]))
    .use(approvalGate)
    .use(agent.tools([runCommand]))
    .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-6" } }));

// ── CLI driver ──────────────────────────────────────────────────────────────
// Usage:
//   pnpm tsx index.ts <session-id> "<message>"
//   pnpm tsx index.ts <session-id> /approve
//   pnpm tsx index.ts <session-id> /deny [reason]
//
// Generate a fresh id with `uuidgen` (macOS/linux) or any UUID generator
// and reuse it across calls. Sessions are persisted in agent.db.

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
    agentId: assistant.agentId,
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
