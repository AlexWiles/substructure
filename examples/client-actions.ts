import Substructure, { contentText, middleware, type RunStream } from "@substructure.ai/sdk";
import type { ClientIdentity, Message, WorkerAction } from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

const TOOL_CATALOG = [
    {
        name: "add",
        description: "Add two numbers",
        parameters: {
            type: "object",
            properties: { a: { type: "number" }, b: { type: "number" } },
            required: ["a", "b"],
        },
    },
    {
        name: "get_weather",
        description: "Get weather by city",
        parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
        },
    },
];

const actionAgent = agent({ id: "action-demo" })
    .use(agent.jsonState())
    .use(
        middleware({
            state: { messages: [] as Message[] },
            handler: (ctx, next) => {
                if (ctx.trigger.type !== "client.action") {
                    return next(ctx);
                }

                switch (ctx.trigger.name) {
                    case "get_tools": {
                        const data = { tools: TOOL_CATALOG, count: TOOL_CATALOG.length };
                        const actions: WorkerAction[] = [{ type: "done", data }];
                        return { actions, state: ctx.state };
                    }

                    case "clear_context": {
                        const previousCount = ctx.state.messages.length;
                        ctx.state.messages = [];
                        const actions: WorkerAction[] = [
                            { type: "done", data: { ok: true, cleared_messages: previousCount } },
                        ];
                        return { actions, state: ctx.state };
                    }

                    default: {
                        const actions: WorkerAction[] = [
                            { type: "done", data: { ok: false, error: `Unknown action: ${ctx.trigger.name}` } },
                        ];
                        return { actions, state: ctx.state };
                    }
                }
            },
        }),
    )
    .use((ctx, next) => {
        if (ctx.trigger.type !== "user.message") {
            return next(ctx);
        }

        ctx.state.messages.push(ctx.trigger.message);
        const actions: WorkerAction[] = [
            {
                type: "done",
                data: {
                    message_count: ctx.state.messages.length,
                    last_message: contentText(ctx.trigger.message.content),
                },
            },
        ];
        return { actions, state: ctx.state };
    });

const embedded = await sub.embedded({ agents: [actionAgent] });

const auth: ClientIdentity = { tenant_id: "default", sub: "example-user" };
const sessionId = crypto.randomUUID();

async function drainToResult(stream: RunStream) {
    for await (const _event of stream) {
    }
    return stream.result;
}

const firstTurn = await drainToResult(
    embedded.submit({
        agentId: "action-demo",
        payload: { type: "message", message: { role: "user", content: "hello there" } },
        sessionId,
        auth,
        turnId: crypto.randomUUID(),
    }),
);
console.log("first run:", firstTurn);

const secondTurn = await drainToResult(
    embedded.submit({
        agentId: "action-demo",
        payload: { type: "message", message: { role: "user", content: "one more message" } },
        sessionId,
        auth,
        turnId: crypto.randomUUID(),
    }),
);
console.log("second run:", secondTurn);

const tools = await drainToResult(
    embedded.submit({
        agentId: "action-demo",
        payload: { type: "action", name: "get_tools" },
        sessionId,
        auth,
    }),
);
console.log("request(get_tools):", tools.data);

const clear = await drainToResult(
    embedded.submit({
        agentId: "action-demo",
        payload: { type: "action", name: "clear_context" },
        sessionId,
        auth,
    }),
);
console.log("request(clear_context):", clear.data);

const afterClear = await drainToResult(
    embedded.submit({
        agentId: "action-demo",
        payload: { type: "message", message: { role: "user", content: "after clear" } },
        sessionId,
        auth,
        turnId: crypto.randomUUID(),
    }),
);
console.log("after clear run:", afterClear);

await embedded.shutdown();
