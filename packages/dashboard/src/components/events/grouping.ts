import type { Event, EventPayload } from "@substructure.ai/sdk/types";

// ── Grouping Rules ──────────────────────────────────────────────────────────

export interface GroupingRule {
    startType: EventPayload["type"];
    endTypes: EventPayload["type"][];
    /** Extract the correlation key from a related payload, or null if unrelated. */
    correlationId: (payload: EventPayload) => string | null;
    label: string;
    summarize: (startPayload: EventPayload, endPayload?: EventPayload) => string;
}

function truncate(s: string, max = 60): string {
    if (s.length <= max) return s;
    return s.slice(0, max) + "\u2026";
}

export const GROUPING_RULES: GroupingRule[] = [
    {
        startType: "llm.call.requested",
        endTypes: ["llm.call.completed", "llm.call.errored"],
        label: "LLM Call",
        correlationId: (p) => {
            switch (p.type) {
                case "llm.call.requested":
                case "llm.call.completed":
                case "llm.call.errored":
                    return p.call_id;
                default:
                    return null;
            }
        },
        summarize: (start, end) => {
            const model = start.type === "llm.call.requested" ? start.request.model : "";
            if (!end) return model;
            if (end.type === "llm.call.completed") {
                const cost = end.response.cost ? `$${end.response.cost}` : "";
                const reason = end.response.finish_reason ?? "";
                return [model, cost, reason].filter(Boolean).join(" \u00b7 ");
            }
            if (end.type === "llm.call.errored") {
                return `${model} \u00b7 ${truncate(end.error)}`;
            }
            return model;
        },
    },
    {
        startType: "worker.decision.requested",
        endTypes: ["worker.decision.completed", "worker.decision.errored"],
        label: "Worker Decision",
        correlationId: (p) => {
            switch (p.type) {
                case "worker.decision.requested":
                case "worker.decision.completed":
                case "worker.decision.errored":
                    return p.decision_id;
                default:
                    return null;
            }
        },
        summarize: (start, end) => {
            const trigger = start.type === "worker.decision.requested" ? start.trigger.type : "";
            if (end?.type === "worker.decision.errored") {
                return `${trigger} \u00b7 ${truncate(end.error)}`;
            }
            return trigger;
        },
    },
    {
        startType: "tool.call.requested",
        endTypes: ["tool.call.completed", "tool.call.errored"],
        label: "Tool Call",
        correlationId: (p) => {
            switch (p.type) {
                case "tool.call.requested":
                case "tool.call.completed":
                case "tool.call.errored":
                    return p.tool_call_id;
                default:
                    return null;
            }
        },
        summarize: (start, end) => {
            const name = start.type === "tool.call.requested" ? start.name : "";
            if (end?.type === "tool.call.errored") {
                return `${name} \u00b7 ${truncate(end.error)}`;
            }
            return name;
        },
    },
    {
        startType: "sub_agent.requested",
        endTypes: ["sub_agent.turn_completed", "sub_agent.errored"],
        label: "Sub-Agent",
        correlationId: (p) => {
            switch (p.type) {
                case "sub_agent.requested":
                case "sub_agent.started":
                case "sub_agent.turn_completed":
                case "sub_agent.errored":
                    return p.session_id;
                default:
                    return null;
            }
        },
        summarize: (start, end) => {
            const agentId = start.type === "sub_agent.requested" ? start.agent_id : "";
            if (end?.type === "sub_agent.errored") {
                return `${agentId} \u00b7 ${truncate(end.error)}`;
            }
            if (end?.type === "sub_agent.turn_completed") {
                return `${agentId} \u00b7 $${end.cost}`;
            }
            return agentId;
        },
    },
];

// ── Types ───────────────────────────────────────────────────────────────────

export interface EventGroup {
    kind: "group";
    rule: GroupingRule;
    startEvent: Event;
    endEvent?: Event;
    innerEvents: Event[];
}

export type GroupedItem = EventGroup | Event;

export function isGroup(item: GroupedItem): item is EventGroup {
    return "kind" in item && item.kind === "group";
}
