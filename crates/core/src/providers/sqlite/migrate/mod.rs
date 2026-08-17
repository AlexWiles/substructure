//! Versioned schema migrations for the tables core owns.
//!
//! `SqliteDb::open` runs these before it returns, so an unmigrated database
//! cannot be observed. A host that keeps its own tables in the same file runs
//! its own list against its own ledger table; the two never interact.
//!
//! Never edit an applied migration — append a new one.

mod m001_baseline;
mod m002_slack_files;
mod m003_blobs;
mod m004_secret_store;
mod m005_auth_flows;

use chrono::Utc;
use rusqlite::{Connection, Transaction, TransactionBehavior};

use crate::event_store::StoreError;

use super::SqliteDb;

/// Ledger table for core's own migrations.
pub const CORE_LEDGER: &str = "substructure_migrations";

pub struct Migration {
    pub version: i64,
    pub name: &'static str,
    pub up: fn(&Connection) -> rusqlite::Result<()>,
}

pub const CORE_MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "baseline",
        up: m001_baseline::up,
    },
    Migration {
        version: 2,
        name: "slack_files",
        up: m002_slack_files::up,
    },
    Migration {
        version: 3,
        name: "blobs",
        up: m003_blobs::up,
    },
    Migration {
        version: 4,
        name: "secret_store",
        up: m004_secret_store::up,
    },
    Migration {
        version: 5,
        name: "auth_flows",
        up: m005_auth_flows::up,
    },
];

/// Applies every pending migration and returns the resulting schema version.
pub fn run(db: &SqliteDb, ledger: &str, migrations: &[Migration]) -> Result<i64, StoreError> {
    let mut conn = db.writer.lock().map_err(internal)?;
    run_on_conn(&mut conn, ledger, migrations)
}

fn run_on_conn(
    conn: &mut Connection,
    ledger: &str,
    migrations: &[Migration],
) -> Result<i64, StoreError> {
    debug_assert!(
        migrations.windows(2).all(|w| w[0].version < w[1].version),
        "migrations must be sorted by strictly increasing version"
    );

    // Immediate takes the write lock before the version is read, so two
    // processes opening the same file cannot both apply the same migration.
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(internal)?;

    tx.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {ledger} (
            version     INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        );"
    ))
    .map_err(internal)?;

    let applied: i64 = tx
        .query_row(
            &format!("SELECT COALESCE(MAX(version), 0) FROM {ledger}"),
            [],
            |row| row.get(0),
        )
        .map_err(internal)?;

    check_applied(&tx, ledger, migrations)?;

    let mut current = applied;
    for migration in migrations.iter().filter(|m| m.version > applied) {
        (migration.up)(&tx).map_err(|e| {
            StoreError::Internal(format!(
                "migration {} ({}): {e}",
                migration.version, migration.name
            ))
        })?;
        tx.execute(
            &format!("INSERT INTO {ledger} (version, name, applied_at) VALUES (?1, ?2, ?3)"),
            rusqlite::params![migration.version, migration.name, Utc::now().to_rfc3339()],
        )
        .map_err(internal)?;
        current = migration.version;
        tracing::info!(
            version = migration.version,
            name = migration.name,
            ledger,
            "applied schema migration"
        );
    }
    tx.commit().map_err(internal)?;
    Ok(current)
}

/// Every applied version must still exist in the list, under the same name. A
/// mismatch means the list was reordered, renumbered, or rolled back, and the
/// schema on disk is not the one this binary expects.
fn check_applied(
    tx: &Transaction,
    ledger: &str,
    migrations: &[Migration],
) -> Result<(), StoreError> {
    let mut stmt = tx
        .prepare(&format!(
            "SELECT version, name FROM {ledger} ORDER BY version"
        ))
        .map_err(internal)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(internal)?;

    for row in rows {
        let (version, name) = row.map_err(internal)?;
        match migrations.iter().find(|m| m.version == version) {
            Some(m) if m.name != name => {
                return Err(StoreError::Internal(format!(
                    "migration {version} was applied as `{name}` but this binary calls it \
                     `{}`; the migration list was reordered or renumbered",
                    m.name
                )))
            }
            None => {
                return Err(StoreError::Internal(format!(
                    "migration {version} (`{name}`) is applied but this binary does not have \
                     it; the database is newer than the binary"
                )))
            }
            _ => {}
        }
    }
    Ok(())
}

fn internal<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Internal(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use uuid::Uuid;

    /// A real temp file, not `:memory:` — `SqliteDb::open` rewrites `:memory:`
    /// to a process-wide shared cache, which would leak between tests.
    struct TestDb {
        db: SqliteDb,
        path: std::path::PathBuf,
    }

    impl Drop for TestDb {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", self.path.display()));
            }
        }
    }

    fn test_db() -> TestDb {
        let path = std::env::temp_dir().join(format!("migrate-{}.db", Uuid::now_v7()));
        let db = SqliteDb::open(path.to_str().unwrap(), Duration::from_secs(5)).unwrap();
        TestDb { db, path }
    }

    fn schema_dump(db: &SqliteDb) -> String {
        let conn = db.writer.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT sql FROM sqlite_master
                  WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                  ORDER BY type, name",
            )
            .unwrap();
        let rows: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        format!("{}\n", rows.join(";\n\n"))
    }

    /// The schema the migrations build must match the committed snapshot. A
    /// diff here means an applied migration was edited instead of a new one
    /// being appended. Rerun with `UPDATE_SCHEMA_SNAPSHOT=1` to accept a change.
    #[test]
    fn schema_matches_the_committed_snapshot() {
        let t = test_db();
        let actual = schema_dump(&t.db);
        let snapshot = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/providers/sqlite/migrate/schema.sql"
        );

        if std::env::var_os("UPDATE_SCHEMA_SNAPSHOT").is_some() {
            std::fs::write(snapshot, &actual).unwrap();
            return;
        }

        assert_eq!(
            actual,
            include_str!("schema.sql"),
            "schema drifted; append a migration, then rerun with UPDATE_SCHEMA_SNAPSHOT=1"
        );
    }

    #[test]
    fn open_migrates_and_reopening_is_a_noop() {
        let t = test_db();
        {
            let conn = t.db.writer.lock().unwrap();
            let applied: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {CORE_LEDGER}"), [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(applied, CORE_MIGRATIONS.len() as i64);
        }

        let version = run(&t.db, CORE_LEDGER, CORE_MIGRATIONS).unwrap();
        assert_eq!(version, CORE_MIGRATIONS.last().unwrap().version);

        let conn = t.db.writer.lock().unwrap();
        let applied: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {CORE_LEDGER}"), [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(applied, CORE_MIGRATIONS.len() as i64);
    }

    /// A second list against a second ledger shares the file without touching
    /// core's, which is what lets a host keep its own tables here.
    #[test]
    fn a_second_ledger_is_independent() {
        fn up(conn: &Connection) -> rusqlite::Result<()> {
            conn.execute_batch("CREATE TABLE host_thing (id TEXT PRIMARY KEY);")
        }
        const HOST: &[Migration] = &[Migration {
            version: 1,
            name: "host_baseline",
            up,
        }];

        let t = test_db();
        assert_eq!(run(&t.db, "host_migrations", HOST).unwrap(), 1);
        assert_eq!(
            run(&t.db, CORE_LEDGER, CORE_MIGRATIONS).unwrap(),
            CORE_MIGRATIONS.last().unwrap().version
        );

        let conn = t.db.writer.lock().unwrap();
        for (ledger, expected) in [
            (CORE_LEDGER, "baseline"),
            ("host_migrations", "host_baseline"),
        ] {
            let name: String = conn
                .query_row(&format!("SELECT name FROM {ledger}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(name, expected);
        }
    }

    /// A credential written before the secret store becomes a referenced,
    /// shared-slot secret — still plaintext until a key appears.
    #[test]
    fn stored_credentials_move_into_the_secret_store() {
        let path = std::env::temp_dir().join(format!("migrate-conv-{}.db", Uuid::now_v7()));
        let mut conn = Connection::open(&path).unwrap();
        run_on_conn(&mut conn, CORE_LEDGER, &CORE_MIGRATIONS[..3]).unwrap();
        conn.execute(
            "INSERT INTO connector_credentials (tenant_id, connection_id, tokens)
             VALUES ('default', 'linear', '{\"token\":\"tok\"}')",
            [],
        )
        .unwrap();

        run_on_conn(&mut conn, CORE_LEDGER, CORE_MIGRATIONS).unwrap();

        let (subject, secret_id): (String, String) = conn
            .query_row(
                "SELECT subject, secret_id FROM connector_credentials
                 WHERE connection_id = 'linear'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(subject, "", "the old row becomes the shared slot");
        let (data, key_version): (String, Option<i64>) = conn
            .query_row(
                "SELECT data, key_version FROM secrets WHERE id = ?1",
                [&secret_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(data, "{\"token\":\"tok\"}");
        assert_eq!(key_version, None);

        drop(conn);
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[test]
    fn a_database_newer_than_the_binary_is_refused() {
        let t = test_db();
        {
            let conn = t.db.writer.lock().unwrap();
            conn.execute(
                &format!("INSERT INTO {CORE_LEDGER} VALUES (99, 'from_the_future', '2026-01-01')"),
                [],
            )
            .unwrap();
        }
        let err = run(&t.db, CORE_LEDGER, CORE_MIGRATIONS).unwrap_err();
        assert!(format!("{err}").contains("newer than the binary"), "{err}");
    }

    #[test]
    fn a_renamed_migration_is_refused() {
        let t = test_db();
        {
            let conn = t.db.writer.lock().unwrap();
            conn.execute(
                &format!("UPDATE {CORE_LEDGER} SET name = 'something_else' WHERE version = 1"),
                [],
            )
            .unwrap();
        }
        let err = run(&t.db, CORE_LEDGER, CORE_MIGRATIONS).unwrap_err();
        assert!(
            format!("{err}").contains("reordered or renumbered"),
            "{err}"
        );
    }
}
