// An AG-UI HttpAgent that can restore a thread's history on connect.
//
// CopilotKit hydrates a thread by calling the agent's `connect()` (fired by
// `<CopilotChat threadId>` on mount), NOT by seeding `initialMessages`. The stock
// HttpAgent leaves `connect()` unimplemented — it throws and CopilotKit swallows
// it, so a self-managed agent shows no history after a refresh.
//
// We implement `connect()` by running the same AG-UI request against the engine's
// `/connect` route, which streams `RUN_STARTED → MESSAGES_SNAPSHOT → RUN_FINISHED`.
// CopilotKit applies the snapshot to its message list (`MESSAGES_SNAPSHOT` merge),
// so the conversation — text and tool calls — comes back on load.

import { HttpAgent } from "@copilotkit/react-core/v2";

type HttpAgentConfig = ConstructorParameters<typeof HttpAgent>[0];
type RunInput = Parameters<HttpAgent["run"]>[0];

export class HistoryHttpAgent extends HttpAgent {
    private readonly connectUrl: string;
    private readonly connectHeaders: Record<string, string>;

    constructor(config: HttpAgentConfig & { connectUrl: string; connectHeaders?: Record<string, string> }) {
        const { connectUrl, connectHeaders, ...rest } = config;
        super(rest);
        this.connectUrl = connectUrl;
        this.connectHeaders = connectHeaders ?? {};
    }

    // Delegate to a throwaway HttpAgent pointed at `/connect` — its `run()` reuses
    // the library's request + SSE-decode pipeline, so we don't touch internals.
    // The input carries the thread id; the engine reads it to pick the session.
    protected connect(input: RunInput) {
        const probeConfig = {
            url: this.connectUrl,
            headers: this.connectHeaders,
            // Bound fetch: HttpAgent calls `this.fetch(...)`, which throws
            // "Illegal invocation" in Firefox with an unbound reference. (`fetch`
            // is a runtime option not surfaced on the v2 config type, hence the cast.)
            fetch: (url: RequestInfo | URL, init?: RequestInit) => fetch(url, init),
        };
        return new HttpAgent(probeConfig as HttpAgentConfig).run(input);
    }
}
