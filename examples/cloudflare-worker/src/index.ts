import Substructure from "@substructure.ai/sdk";
import { AgentState, durableObjectState } from "./state";

const sub = new Substructure();
const { agent } = sub;

export { AgentState };

let workerEnv: WorkerEnv;

const retry = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const fetchUrl = agent.tool({
    description: "Fetch a URL and return its content. Use for APIs, web pages, or checking if a site is up.",
    parameters: {
        type: "object",
        properties: {
            url: { type: "string", description: "The URL to fetch" },
            max_length: {
                type: "number",
                description: "Max characters to return (default 5000)",
            },
        },
        required: ["url"],
    },
    execute: async (args: string) => {
        const { url, max_length = 5000 } = JSON.parse(args);
        const resp = await fetch(url);
        const text = await resp.text();
        return {
            status: resp.status,
            content_type: resp.headers.get("content-type"),
            body: text.slice(0, max_length),
            truncated: text.length > max_length,
        };
    },
    retry,
});

const extractFromHtml = agent.tool({
    description:
        "Fetch a webpage and extract text content matching a CSS selector. " +
        "Returns an array of matched text strings.",
    parameters: {
        type: "object",
        properties: {
            url: { type: "string", description: "The URL to fetch" },
            selector: {
                type: "string",
                description: "CSS selector to match elements (e.g. 'h1', '.price', 'article p')",
            },
        },
        required: ["url", "selector"],
    },
    execute: async (args: string) => {
        const { url, selector } = JSON.parse(args);
        const resp = await fetch(url);
        const matches: string[] = [];
        let current = "";

        const rewriter = new HTMLRewriter().on(selector, {
            element() {
                current = "";
            },
            text(chunk) {
                current += chunk.text;
                if (chunk.lastInTextNode) {
                    const trimmed = current.trim();
                    if (trimmed) matches.push(trimmed);
                    current = "";
                }
            },
        });

        await rewriter.transform(resp).text();
        return { url, selector, matches };
    },
    retry,
});

const SYSTEM_PROMPT =
    "You are a web research assistant running on Cloudflare Workers. " +
    "You can fetch URLs and extract content from web pages using CSS selectors. " +
    "Be concise and summarize what you find.";

const webAgent = agent({ id: "web-agent" })
    .use(agent.logging())
    .use(durableObjectState(() => workerEnv.AGENT_STATE))
    .use(agent.messageHistory())
    .use(agent.systemMessage(SYSTEM_PROMPT))
    .use(agent.tools({ fetchUrl, extractFromHtml }))
    .use(
        agent.llmLoop({
            request: { model: "arcee-ai/trinity-large-preview:free" },
            llm_client: "openrouter",
            retry,
        }),
    );

// Register this worker with the backend via the API:
//
//   backend.registerWorker({
//     transport_type: "http",
//     config: { endpoint_url: "https://your-worker.workers.dev" },
//   });
//
// Or start the server with: --worker-url https://your-worker.workers.dev

const worker = sub.worker({ agents: [webAgent] });

interface WorkerEnv extends Env {
    SIGNING_SECRET?: string;
}

export default {
    async fetch(request: Request, env: WorkerEnv): Promise<Response> {
        workerEnv = env;
        const handler = worker.fetchHandler({
            signingSecret: env.SIGNING_SECRET,
        });
        try {
            return await handler(request);
        } catch (err) {
            const message = err instanceof Error ? err.message : "Internal error";
            return Response.json({ error: message }, { status: 500 });
        }
    },
};
