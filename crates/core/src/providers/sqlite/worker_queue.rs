use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use tokio::sync::Notify;

use crate::event_store::StoreError;
use crate::worker::{DequeueFilter, WorkerDecisionRequest, WorkerQueue};

use super::SqliteDb;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS worker_queue (
    decision_id    TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    agent_id       TEXT NOT NULL,
    payload        TEXT NOT NULL,
    enqueued_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_worker_queue_tenant_agent_order
    ON worker_queue (tenant_id, agent_id, enqueued_at, decision_id);
";

const DEQUEUE_POLL_INTERVAL: Duration = Duration::from_millis(500);

pub struct SqliteWorkerQueue {
    db: SqliteDb,
    notify: Notify,
}

impl SqliteWorkerQueue {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self {
            db,
            notify: Notify::new(),
        })
    }
}

#[async_trait]
impl WorkerQueue for SqliteWorkerQueue {
    async fn enqueue(&self, decision: WorkerDecisionRequest) {
        let writer = self.db.writer.clone();
        let result = tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            do_enqueue(&conn, decision)
        })
        .await;

        match result {
            Ok(Ok(())) => self.notify.notify_waiters(),
            Ok(Err(err)) => {
                tracing::error!(error = %err, "failed to enqueue worker decision");
            }
            Err(err) => {
                tracing::error!(error = %err, "worker queue enqueue task failed");
            }
        }
    }

    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest> {
        loop {
            let notified = self.notify.notified();
            let writer = self.db.writer.clone();
            let tenant_id = filter.tenant_id.clone();
            let result = tokio::task::spawn_blocking(move || {
                let mut conn = writer.lock().map_err(|e| e.to_string())?;
                do_dequeue(&mut conn, &tenant_id)
            })
            .await;

            match result {
                Ok(Ok(Some(decision))) => return Some(decision),
                Ok(Ok(None)) => {}
                Ok(Err(err)) => {
                    tracing::error!(error = %err, "failed to dequeue worker decision");
                }
                Err(err) => {
                    tracing::error!(error = %err, "worker queue dequeue task failed");
                }
            }

            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(DEQUEUE_POLL_INTERVAL) => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn do_enqueue(conn: &Connection, decision: WorkerDecisionRequest) -> Result<(), String> {
    let payload = serde_json::to_string(&decision).map_err(|e| e.to_string())?;
    let enqueued_at = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO worker_queue (decision_id, tenant_id, agent_id, payload, enqueued_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(decision_id) DO UPDATE SET
            tenant_id = excluded.tenant_id,
            agent_id = excluded.agent_id,
            payload = excluded.payload,
            enqueued_at = excluded.enqueued_at",
        rusqlite::params![
            decision.decision_id,
            decision.tenant_id,
            decision.agent_id,
            payload,
            enqueued_at,
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn do_dequeue(
    conn: &mut Connection,
    tenant_id: &str,
) -> Result<Option<WorkerDecisionRequest>, String> {
    let tx = conn.transaction().map_err(|e| e.to_string())?;

    let row = tx
        .query_row(
            "SELECT decision_id, payload
             FROM worker_queue
             WHERE tenant_id = ?1
             ORDER BY enqueued_at ASC, decision_id ASC
             LIMIT 1",
            rusqlite::params![tenant_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|e| e.to_string())?;

    let Some((decision_id, payload)) = row else {
        tx.commit().map_err(|e| e.to_string())?;
        return Ok(None);
    };

    tx.execute(
        "DELETE FROM worker_queue WHERE decision_id = ?1",
        rusqlite::params![decision_id],
    )
    .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;

    let decision =
        serde_json::from_str::<WorkerDecisionRequest>(&payload).map_err(|e| e.to_string())?;
    Ok(Some(decision))
}
