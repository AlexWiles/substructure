// Human-in-the-loop tool approval: a gated tool pauses the session with an
// approval prompt instead of running; the resolution decides what happens.
import { serve } from "@hono/node-server";
import { Hono } from "hono";

const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    },
    {
        name: "send_email",
        description: "Send an email.",
        input: {
            type: "object",
            properties: {
                to: { type: "string" },
                subject: { type: "string" },
                body: { type: "string" }
            },
            required: ["to", "subject", "body"]
        },
        approval: true,
        exec: ({ to, subject }) => `Sent "${subject}" to ${to}.`
    }
];

// The payload is an AG-UI Interrupt: spec fields top-level, everything
// convention-specific under `metadata` (client-visible — secrets belong in
// worker state). Slack posts `message` with `metadata.options` as buttons; a
// click resumes with the chosen option's value.
const approvalInterrupt = (trigger) => ({
    type: "interrupt",
    interrupt_id: `approve:${trigger.id}`,
    reason: "tool_call",
    payload: {
        message: `Run \`${trigger.name}\`?\n\`\`\`\n${JSON.stringify(trigger.input.value, null, 2)}\n\`\`\``,
        toolCallId: trigger.id,
        metadata: {
            options: [
                { label: "Approve", value: { decision: "approve" }, style: "primary" },
                { label: "Deny", value: { decision: "deny" }, style: "danger" }
            ],
            pending: {
                tool_call: { id: trigger.id, name: trigger.name, arguments: trigger.input.value }
            }
        }
    }
});

function decide({ trigger, proposed, state }) {
    state = state ?? { pending: {} };

    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map(({ name, description, input }) => ({ name, description, input }))
            },
            state
        };
    }

    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        if (tool.approval) {
            // Hold the call: no tool.result, just the interrupt. The engine
            // keeps the call pending until a resume settles it.
            state.pending[`approve:${trigger.id}`] = {
                id: trigger.id,
                name: trigger.name,
                args: trigger.input.value
            };
            return { state, actions: [approvalInterrupt(trigger)] };
        }
        return { actions: [{ type: "tool.result", result: tool.exec(trigger.input.value) }] };
    }

    if (trigger.type === "interrupt.resumed") {
        const held = state.pending[trigger.interrupt_id];
        if (!held) return proposed;
        delete state.pending[trigger.interrupt_id];
        // The AG-UI resume shape: { status, payload, responder }. Anything
        // but an explicit resolved approve denies — a resume can carry any payload.
        const { status, payload, responder } = trigger.payload ?? {};
        if (status !== "resolved" || payload?.decision !== "approve") {
            const by = responder?.user ? ` by <@${responder.user}>` : " by the user";
            return {
                state,
                actions: [{ type: "tool.result", id: held.id, result: `Denied${by}.` }]
            };
        }
        const tool = tools.find((t) => t.name === held.name);
        return {
            state,
            actions: [{ type: "tool.result", id: held.id, result: tool.exec(held.args) }]
        };
    }

    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
