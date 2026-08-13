//! Attachment and generated-media bytes, keyed like every blob store:
//! `(tenant_id, id)`, one row per put.

use rusqlite::Connection;

const SQL: &str = "
CREATE TABLE blobs (
    tenant_id  TEXT NOT NULL,
    id         TEXT NOT NULL,
    mime       TEXT NOT NULL,
    name       TEXT,
    size       INTEGER NOT NULL,
    bytes      BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id)
);
";

pub fn up(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SQL)
}
