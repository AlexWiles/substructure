use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use tokio::sync::{oneshot, Mutex};
use tokio::sync::Notify;

use crate::worker::{DequeueFilter, WorkerDecisionRequest, WorkerQueue};

struct Waiter {
    tenant_id: String,
    agent_ids: Vec<String>,
    tx: oneshot::Sender<WorkerDecisionRequest>,
}

impl Waiter {
    fn matches(&self, decision: &WorkerDecisionRequest) -> bool {
        self.tenant_id == decision.tenant_id
            && self.agent_ids.iter().any(|id| id == &decision.agent_id)
    }
}

struct Inner {
    items: VecDeque<WorkerDecisionRequest>,
    waiters: VecDeque<Waiter>,
}

pub struct InMemoryWorkerQueue {
    inner: Mutex<Inner>,
}

impl InMemoryWorkerQueue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                items: VecDeque::new(),
                waiters: VecDeque::new(),
            }),
        }
    }
}

fn matches_filter(decision: &WorkerDecisionRequest, filter: &DequeueFilter) -> bool {
    decision.tenant_id == filter.tenant_id
        && filter.agent_ids.iter().any(|id| id == &decision.agent_id)
}

#[async_trait]
impl WorkerQueue for InMemoryWorkerQueue {
    async fn enqueue(&self, mut decision: WorkerDecisionRequest) {
        let mut inner = self.inner.lock().await;
        // Remove stale waiters and try to hand off to a live one
        inner.waiters.retain(|w| !w.tx.is_closed());
        while let Some(idx) = inner.waiters.iter().position(|w| w.matches(&decision)) {
            let waiter = inner.waiters.remove(idx).unwrap();
            match waiter.tx.send(decision) {
                Ok(()) => return,
                Err(d) => decision = d,
            }
        }
        inner.items.push_back(decision);
    }

    async fn dequeue(&self, filter: &DequeueFilter) -> Option<WorkerDecisionRequest> {
        let rx = {
            let mut inner = self.inner.lock().await;
            // Check buffered items first
            if let Some(idx) = inner.items.iter().position(|d| matches_filter(d, filter)) {
                return inner.items.remove(idx);
            }
            // No match — register as a waiter
            let (tx, rx) = oneshot::channel();
            inner.waiters.push_back(Waiter {
                tenant_id: filter.tenant_id.clone(),
                agent_ids: filter.agent_ids.clone(),
                tx,
            });
            rx
        };
        rx.await.ok()
    }
}

const SQLITE_WORKER_QUEUE_SCHEMA: &str = "
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

const DEFAULT_DEQUEUE_POLL_INTERVAL: Duration = Duration::from_millis(500);

pub struct SqliteWorkerQueue {
    writer: Arc<StdMutex<Connection>>,
    notify: Notify,
    poll_interval: Duration,
}

impl SqliteWorkerQueue {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::with_poll_interval(path, DEFAULT_DEQUEUE_POLL_INTERVAL)
    }

    pub fn with_poll_interval(path: impl AsRef<Path>, poll_interval: Duration) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        conn.execute_batch(SQLITE_WORKER_QUEUE_SCHEMA)
            .map_err(|e| e.to_string())?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            writer: Arc::new(StdMutex::new(conn)),
            notify: Notify::new(),
            poll_interval,
        })
    }

    async fn dequeue_once(&self, filter: &DequeueFilter) -> Result<Option<WorkerDecisionRequest>, String> {
        if filter.agent_ids.is_empty() {
            return Ok(None);
        }

        let writer = self.writer.clone();
        let tenant_id = filter.tenant_id.clone();
        let agent_ids = filter.agent_ids.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = writer.lock().map_err(|e| e.to_string())?;
            sqlite_dequeue(&mut conn, &tenant_id, &agent_ids)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

fn sqlite_enqueue(conn: &Connection, decision: WorkerDecisionRequest) -> Result<(), String> {
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

fn sqlite_dequeue(
    conn: &mut Connection,
    tenant_id: &str,
    agent_ids: &[String],
) -> Result<Option<WorkerDecisionRequest>, String> {
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let placeholders = std::iter::repeat("?")
        .take(agent_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let select_sql = format!(
        "SELECT decision_id, payload
         FROM worker_queue
         WHERE tenant_id = ? AND agent_id IN ({placeholders})
         ORDER BY enqueued_at ASC, decision_id ASC
         LIMIT 1"
    );

    let mut params: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(agent_ids.len() + 1);
    params.push(&tenant_id);
    for agent_id in agent_ids {
        params.push(agent_id);
    }

    let row = tx
        .query_row(&select_sql, rusqlite::params_from_iter(params), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
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

#[async_trait]
impl WorkerQueue for SqliteWorkerQueue {
    async fn enqueue(&self, decision: WorkerDecisionRequest) {
        let writer = self.writer.clone();
        let result = tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            sqlite_enqueue(&conn, decision)
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
        if filter.agent_ids.is_empty() {
            return None;
        }

        loop {
            let notified = self.notify.notified();
            match self.dequeue_once(filter).await {
                Ok(Some(decision)) => return Some(decision),
                Ok(None) => {}
                Err(err) => {
                    tracing::error!(error = %err, "failed to dequeue worker decision");
                }
            }

            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(self.poll_interval) => {}
            }
        }
    }
}
