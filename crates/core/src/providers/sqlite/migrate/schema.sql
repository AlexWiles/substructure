CREATE INDEX idx_engine_worker_queue_tenant_agent_order
    ON engine_worker_queue (tenant_id, agent_id, enqueued_at, decision_id);

CREATE INDEX idx_events_session ON events (tenant_id, session_id);

CREATE UNIQUE INDEX idx_events_session_seq ON events (tenant_id, session_id, seq);

CREATE INDEX idx_messages_session ON messages (tenant_id, session_id, seq);

CREATE INDEX idx_session_index_last_event
  ON session_index (tenant_id, last_event_at DESC, session_id DESC);

CREATE INDEX idx_session_index_top_level_last_event
  ON session_index (tenant_id, top_level, last_event_at DESC, session_id DESC);

CREATE INDEX idx_session_index_wake_at
  ON session_index (tenant_id, wake_at);

CREATE INDEX idx_snapshots_last_event ON snapshots (last_event_at);

CREATE INDEX idx_snapshots_wake_at ON snapshots (wake_at);

CREATE INDEX idx_wake_schedule_wake_at ON wake_schedule (wake_at);

CREATE TABLE connector_credentials (
    tenant_id     TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    tokens        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, connection_id)
);

CREATE TABLE engine_worker_queue (
    decision_id  TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    payload      TEXT NOT NULL,
    enqueued_at  TEXT NOT NULL
);

CREATE TABLE events (
    id               TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    occurred_at      TEXT NOT NULL,
    start_time       TEXT NOT NULL,
    end_time         TEXT NOT NULL,
    span             TEXT NOT NULL,
    payload          TEXT NOT NULL,
    meta             TEXT NOT NULL
);

CREATE TABLE llm_prompts (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    call_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, call_id)
);

CREATE TABLE messages (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    parent_id   TEXT,
    seq         INTEGER NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, message_id)
);

CREATE TABLE projection_cursors (
    projection_name TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    owner_id        TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (projection_name, tenant_id, session_id)
);

CREATE TABLE projection_seeds (
    projection_name TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL
);

CREATE TABLE session_index (
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    first_event_at  TEXT,
    last_event_at   TEXT,
    wake_at         TEXT,
    top_level       INTEGER NOT NULL,
    agent_id        TEXT,
    cost            TEXT NOT NULL,
    sub_agent_cost  TEXT NOT NULL,
    status_json     TEXT NOT NULL,
    turn_id         TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id)
);

CREATE TABLE session_versions (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    anchor      TEXT,
    data        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, kind, seq)
);

CREATE TABLE slack_files (
    tenant_id  TEXT NOT NULL,
    blob_id    TEXT NOT NULL,
    file_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, blob_id)
);

CREATE TABLE slack_turn_streams (
    tenant_id  TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id    TEXT NOT NULL,
    start_seq  INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ts         TEXT,
    version    INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id, turn_id)
);

CREATE TABLE snapshots (
    tenant_id       TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    shard_key       INTEGER NOT NULL,
    data            TEXT NOT NULL,
    wake_at         TEXT,
    first_event_at  TEXT,
    last_event_at   TEXT,
    PRIMARY KEY (tenant_id, session_id)
);

CREATE TABLE substructure_migrations (
            version     INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        );

CREATE TABLE wake_schedule (
    tenant_id   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    wake_at     TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id)
)
