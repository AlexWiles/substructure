pub mod checkpoint;
pub mod event_store;
pub mod push;
pub mod session_index;
pub mod wake;
pub mod worker_queue;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags};

use crate::event_store::StoreError;

pub use checkpoint::SqliteCheckpointStore;
pub use event_store::SqliteEventStore;
pub use push::SqlitePushStore;
pub use session_index::SqliteSessionIndexStore;
pub use wake::SqliteWakeStore;
pub use worker_queue::SqliteWorkerQueue;

#[derive(Clone)]
pub struct ReaderConfig {
    path: String,
    busy_timeout: Duration,
}

impl ReaderConfig {
    pub fn open(&self) -> Result<Connection, StoreError> {
        let conn = Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| StoreError::Internal(e.to_string()))?;
        conn.busy_timeout(self.busy_timeout)
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        Ok(conn)
    }
}

/// A shared SQLite connection pair (writer + reader config).
///
/// Clone is cheap — both fields are behind Arc / small data.
/// Each store struct holds one of these.
#[derive(Clone)]
pub struct SqliteDb {
    pub writer: Arc<Mutex<Connection>>,
    pub reader: ReaderConfig,
}

impl SqliteDb {
    /// Opens a SQLite database at `path` and configures pragmas.
    /// Does **not** run any schema — each store does that in its own `new()`.
    pub fn open(path: &str, busy_timeout: Duration) -> Result<Self, StoreError> {
        let in_memory = path == ":memory:";

        let path_str = if in_memory {
            "file::memory:?mode=memory&cache=shared".to_string()
        } else {
            path.to_string()
        };

        let open_flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_URI;

        let writer = Connection::open_with_flags(&path_str, open_flags)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        writer
            .busy_timeout(busy_timeout)
            .map_err(|e| StoreError::Internal(e.to_string()))?;

        if !in_memory {
            writer
                .pragma_update(None, "journal_mode", "WAL")
                .map_err(|e| StoreError::Internal(e.to_string()))?;
        }

        let reader = ReaderConfig {
            path: path_str,
            busy_timeout,
        };

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            reader,
        })
    }

    /// Runs an idempotent DDL schema string on the writer connection.
    pub fn run_schema(&self, schema: &str) -> Result<(), StoreError> {
        let conn = self
            .writer
            .lock()
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        conn.execute_batch(schema)
            .map_err(|e| StoreError::Internal(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) fn parse_dt(s: &str) -> Option<DateTime<Utc>> {
    s.parse().ok()
}

pub(crate) fn spawn_err(e: tokio::task::JoinError) -> StoreError {
    StoreError::Internal(format!("spawn_blocking: {e}"))
}

/// Convert sea-query Values to rusqlite params.
pub(crate) fn sea_params(values: sea_query::Values) -> Vec<Box<dyn rusqlite::types::ToSql>> {
    values
        .0
        .into_iter()
        .map(|v| -> Box<dyn rusqlite::types::ToSql> {
            match v {
                sea_query::Value::String(s) => Box::new(s),
                sea_query::Value::Int(i) => Box::new(i),
                sea_query::Value::BigInt(i) => Box::new(i),
                sea_query::Value::BigUnsigned(u) => Box::new(u.map(|n| n as i64)),
                sea_query::Value::Bool(b) => Box::new(b),
                _ => Box::new(Option::<String>::None),
            }
        })
        .collect()
}
