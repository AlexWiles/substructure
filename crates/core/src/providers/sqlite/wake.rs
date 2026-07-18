use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension};

use crate::event_store::StoreError;
use crate::wake::{WakeScheduleItem, WakeScheduleStore};

use super::{parse_dt, SqliteDb};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS wake_schedule (
    tenant_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    wake_at         TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_wake_schedule_wake_at ON wake_schedule (wake_at);
";

pub struct SqliteWakeStore {
    db: SqliteDb,
}

impl SqliteWakeStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }
}

#[async_trait]
impl WakeScheduleStore for SqliteWakeStore {
    async fn upsert_wake(
        &self,
        tenant_id: &str,
        session_id: &str,
        wake_at: DateTime<Utc>,
    ) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let session_id = session_id.to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_upsert_wake(&conn, &tenant_id, &session_id, wake_at)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn remove_wake(&self, tenant_id: &str, session_id: &str) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let session_id = session_id.to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_remove_wake(&conn, &tenant_id, &session_id)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn list_due_wakes(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<WakeScheduleItem>, String> {
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(|e| e.to_string())?;
            do_list_due_wakes(&conn, now, limit)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn next_wake_at(&self) -> Result<Option<DateTime<Utc>>, String> {
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(|e| e.to_string())?;
            do_next_wake_at(&conn)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn do_upsert_wake(
    conn: &Connection,
    tenant_id: &str,
    session_id: &str,
    wake_at: DateTime<Utc>,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO wake_schedule (tenant_id, session_id, wake_at, updated_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(tenant_id, session_id) DO UPDATE SET
             wake_at = excluded.wake_at,
             updated_at = excluded.updated_at",
        rusqlite::params![
            tenant_id,
            session_id,
            wake_at.to_rfc3339(),
            Utc::now().to_rfc3339()
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn do_remove_wake(conn: &Connection, tenant_id: &str, session_id: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM wake_schedule WHERE tenant_id = ?1 AND session_id = ?2",
        rusqlite::params![tenant_id, session_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn do_list_due_wakes(
    conn: &Connection,
    now: DateTime<Utc>,
    limit: usize,
) -> Result<Vec<WakeScheduleItem>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT tenant_id, session_id, wake_at
             FROM wake_schedule
             WHERE wake_at <= ?1
             ORDER BY wake_at ASC
             LIMIT ?2",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map(rusqlite::params![now.to_rfc3339(), limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for row in rows {
        let (tenant_id, session_id, wake_at_str) = row.map_err(|e| e.to_string())?;
        let Some(wake_at) = parse_dt(&wake_at_str) else {
            continue;
        };
        out.push(WakeScheduleItem {
            tenant_id,
            session_id,
            wake_at,
        });
    }
    Ok(out)
}

fn do_next_wake_at(conn: &Connection) -> Result<Option<DateTime<Utc>>, String> {
    let wake_at = conn
        .query_row(
            "SELECT wake_at FROM wake_schedule ORDER BY wake_at ASC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    Ok(wake_at.as_deref().and_then(parse_dt))
}
