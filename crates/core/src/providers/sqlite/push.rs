use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::event_store::StoreError;
use crate::worker::push::{PushRegistrationRecord, PushRegistrationStore};

use super::SqliteDb;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS push_registrations (
    tenant_id       TEXT PRIMARY KEY,
    transport_type  TEXT NOT NULL,
    config          TEXT NOT NULL
);
";

pub struct SqlitePushStore {
    db: SqliteDb,
}

impl SqlitePushStore {
    pub fn new(db: SqliteDb) -> Result<Self, StoreError> {
        db.run_schema(SCHEMA)?;
        Ok(Self { db })
    }
}

#[async_trait]
impl PushRegistrationStore for SqlitePushStore {
    async fn save(&self, record: &PushRegistrationRecord) -> Result<(), String> {
        let record = record.clone();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            let config_str = serde_json::to_string(&record.config).map_err(|e| e.to_string())?;
            conn.execute(
                "INSERT INTO push_registrations (tenant_id, transport_type, config)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(tenant_id) DO UPDATE SET
                     transport_type = excluded.transport_type,
                     config = excluded.config",
                rusqlite::params![record.tenant_id, record.transport_type, config_str],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn remove(&self, tenant_id: &str) -> Result<(), String> {
        let tenant_id = tenant_id.to_string();
        let writer = self.db.writer.clone();
        tokio::task::spawn_blocking(move || {
            let conn = writer.lock().map_err(|e| e.to_string())?;
            conn.execute(
                "DELETE FROM push_registrations WHERE tenant_id = ?1",
                rusqlite::params![tenant_id],
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn get(&self, tenant_id: &str) -> Result<Option<PushRegistrationRecord>, String> {
        let tenant_id = tenant_id.to_string();
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(|e| e.to_string())?;
            let result = conn
                .query_row(
                    "SELECT transport_type, config FROM push_registrations
                     WHERE tenant_id = ?1",
                    rusqlite::params![tenant_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()
                .map_err(|e| e.to_string())?;
            match result {
                Some((transport_type, config_str)) => {
                    let config = serde_json::from_str(&config_str).map_err(|e| e.to_string())?;
                    Ok(Some(PushRegistrationRecord {
                        tenant_id,
                        transport_type,
                        config,
                    }))
                }
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| e.to_string())?
    }

    async fn list_tenants(&self) -> Result<Vec<String>, String> {
        let reader = self.db.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.open().map_err(|e| e.to_string())?;
            let mut stmt = conn
                .prepare("SELECT tenant_id FROM push_registrations")
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| e.to_string())?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.map_err(|e| e.to_string())?);
            }
            Ok(result)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}
