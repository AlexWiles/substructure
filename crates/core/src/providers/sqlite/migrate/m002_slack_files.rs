//! One row per blob a workspace has uploaded to Slack: the reply embeds the
//! Slack file id, and a retry or repeat reuses it instead of uploading again.

use rusqlite::Connection;

const SQL: &str = "
CREATE TABLE slack_files (
    tenant_id  TEXT NOT NULL,
    blob_id    TEXT NOT NULL,
    file_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, blob_id)
);
";

pub fn up(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SQL)
}
