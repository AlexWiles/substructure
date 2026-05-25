import { BaseClient, type RequestOptions } from "./base";
import type { Uuid, DateTime, Decimal, Event, SessionState, SessionStatus } from "./types";

// ── Admin response types ────────────────────────────────────────────────────

export interface AggregateSummary {
    aggregate_id: Uuid;
    aggregate_type: string;
    tenant_id: string;
    wake_at?: DateTime;
    stream_version: number;
    first_event_at?: DateTime;
    last_event_at?: DateTime;
}

export interface SessionListItem {
    session_id: Uuid;
    tenant_id: string;
    stream_version: number;
    first_event_at?: DateTime;
    last_event_at?: DateTime;
    wake_at?: DateTime;
    top_level: boolean;
    agent_id?: string;
    cost: Decimal;
    sub_agent_cost: Decimal;
    status: SessionStatus;
    turn_id?: string;
}

export interface SessionListResponse {
    items: SessionListItem[];
    next_cursor?: string;
}

export interface SessionDetail {
    stream_version: number;
    first_event_at?: DateTime;
    last_event_at?: DateTime;
    state: SessionState;
}

// ── Query params ────────────────────────────────────────────────────────────

export type AggregateSort = "last_event_desc" | "first_event_asc" | "first_event_desc" | "wake_at_asc";

export interface ListSessionsParams {
    tenant_id?: string;
    top_level?: boolean;
    sort?: AggregateSort;
    limit?: number;
    cursor?: string;
}

export interface SessionEventsParams {
    sequence_after?: number;
    limit?: number;
}

// ── Client ──────────────────────────────────────────────────────────────────

export class AdminClient extends BaseClient {
    async listSessions(params?: ListSessionsParams): Promise<SessionListResponse> {
        return this.get("/api/admin/sessions", {
            tenant_id: params?.tenant_id,
            top_level: params?.top_level?.toString(),
            sort: params?.sort,
            limit: params?.limit?.toString(),
            cursor: params?.cursor,
        });
    }

    async getSession(sessionId: Uuid): Promise<SessionDetail> {
        return this.get(`/api/admin/sessions/${sessionId}`);
    }

    async getSessionEvents(sessionId: Uuid, params?: SessionEventsParams): Promise<Event[]> {
        return this.get(`/api/admin/sessions/${sessionId}/events`, {
            sequence_after: params?.sequence_after?.toString(),
            limit: params?.limit?.toString(),
        });
    }

    async *streamSessionEvents(
        sessionId: Uuid,
        params?: SessionEventsParams,
        opts?: RequestOptions,
    ): AsyncGenerator<Event> {
        yield* this.streamSSEGet<Event>(
            `/api/admin/sessions/${sessionId}/events/stream`,
            {
                sequence_after: params?.sequence_after?.toString(),
                limit: params?.limit?.toString(),
            },
            opts,
        );
    }
}
