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
            decision.tenant_id(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DecisionTrigger, LlmRequest, SessionOwner};
    use crate::runtime::span::SpanContext;
    use uuid::Uuid;

    fn sample_decision(tenant_id: &str, decision_id: &str) -> WorkerDecisionRequest {
        WorkerDecisionRequest {
            session_id: "sess-1".to_string(),
            decision_id: decision_id.to_string(),
            agent_id: "agent-1".to_string(),
            identity: SessionOwner {
                tenant_id: tenant_id.to_string(),
                id: Some("user-1".to_string()),
                metadata: Default::default(),
            },
            trigger: DecisionTrigger::LlmExecute {
                id: "llm-1".to_string(),
                request: LlmRequest {
                    model: "test-model".to_string(),
                    messages: vec![],
                    tools: None,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: true,
                attempt: 0,
                deadline: None,
            },
            proposed: None,
            state: Default::default(),
            agent: None,
            calls: Default::default(),
            pending_calls: 0,
            transcript: vec![],
            message_tree: Default::default(),
            ancestry: vec![],
            span: SpanContext::root(),
            attempts: 0,
            deadline: None,
            turn_id: None,
        }
    }

    /// The durable queue serializes the decision into SQLite and reads it back.
    /// The tenant lives only on `identity` now, so this proves it survives the
    /// round-trip — otherwise the push loop's `registry.lookup(decision.tenant_id())`
    /// would key on an empty tenant and silently drop the decision.
    #[tokio::test]
    async fn dequeue_preserves_tenant_across_serialization() {
        let path = std::env::temp_dir().join(format!("core-worker-queue-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        let queue = SqliteWorkerQueue::new(db).unwrap();

        queue.enqueue(sample_decision("tenant-xyz", "dec-1")).await;
        let dequeued = queue
            .dequeue(&DequeueFilter {
                tenant_id: "tenant-xyz".to_string(),
            })
            .await
            .expect("a decision");

        assert_eq!(dequeued.tenant_id(), "tenant-xyz");
        assert_eq!(dequeued.decision_id, "dec-1");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }
}
