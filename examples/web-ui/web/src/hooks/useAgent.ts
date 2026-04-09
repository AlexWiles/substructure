import { useCallback, useReducer, useRef } from "react";
import { FrontendClient } from "@substructure.ai/sdk";
import type { Event } from "@substructure.ai/sdk";

interface TurnResult {
    turnId: string;
    data: unknown;
    cost: string;
    tokenUsage: Record<string, number>;
}

interface AgentState {
    status: "idle" | "running" | "completed" | "error";
    events: Event[];
    result: TurnResult | null;
    error: string | null;
}

type Action =
    | { type: "START" }
    | { type: "EVENT"; event: Event }
    | { type: "COMPLETE"; result: TurnResult }
    | { type: "ERROR"; error: string }
    | { type: "RESET" };

function reducer(state: AgentState, action: Action): AgentState {
    switch (action.type) {
        case "START":
            return { status: "running", events: [], result: null, error: null };
        case "EVENT":
            return { ...state, events: [...state.events, action.event] };
        case "COMPLETE":
            return { ...state, status: "completed", result: action.result };
        case "ERROR":
            return { ...state, status: "error", error: action.error };
        case "RESET":
            return { status: "idle", events: [], result: null, error: null };
    }
}

const initialState: AgentState = {
    status: "idle",
    events: [],
    result: null,
    error: null,
};

interface Config {
    serverUrl: string;
    agentId: string;
}

export function useAgent() {
    const [state, dispatch] = useReducer(reducer, initialState);
    const configRef = useRef<Config | null>(null);
    const tokenRef = useRef<string | null>(null);

    const submit = useCallback(async (topic: string) => {
        dispatch({ type: "START" });

        try {
            // Fetch config if we don't have it
            if (!configRef.current) {
                const res = await fetch("/api/config");
                configRef.current = await res.json();
            }

            // Fetch a fresh JWT token
            const tokenRes = await fetch("/api/token");
            const { token } = await tokenRes.json();
            tokenRef.current = token;

            const frontend = new FrontendClient({
                url: configRef.current!.serverUrl,
                token,
            });

            const stream = frontend.submit({
                agentId: configRef.current!.agentId,
                payload: {
                    type: "message",
                    message: { role: "user" as const, content: topic },
                },
            });

            for await (const event of stream) {
                dispatch({ type: "EVENT", event });
            }

            const result = await stream.result;
            dispatch({ type: "COMPLETE", result });
        } catch (err) {
            dispatch({
                type: "ERROR",
                error: err instanceof Error ? err.message : String(err),
            });
        }
    }, []);

    const reset = useCallback(() => {
        dispatch({ type: "RESET" });
    }, []);

    return { ...state, submit, reset };
}
