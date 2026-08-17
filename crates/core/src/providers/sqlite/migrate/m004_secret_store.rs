//! Secrets move into one table other rows reference, and credentials gain a
//! holder: one row per (connection, subject), the shared slot included.
//! Existing credentials become referenced secrets, plaintext (`key_version`
//! NULL) until a key appears and the startup sweep seals them.

use rusqlite::Connection;

pub fn up(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE secrets (
            tenant_id   TEXT NOT NULL,
            id          TEXT NOT NULL,
            data        TEXT NOT NULL,
            key_version INTEGER,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (tenant_id, id)
        );
        CREATE TABLE connector_credentials_v2 (
            tenant_id     TEXT NOT NULL,
            connection_id TEXT NOT NULL,
            subject       TEXT NOT NULL,
            secret_id     TEXT NOT NULL,
            PRIMARY KEY (tenant_id, connection_id, subject)
        );",
    )?;

    let rows: Vec<(String, String, String)> = {
        let mut stmt =
            conn.prepare("SELECT tenant_id, connection_id, tokens FROM connector_credentials")?;
        let mapped = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        mapped.collect::<rusqlite::Result<_>>()?
    };
    let now = chrono::Utc::now().to_rfc3339();
    for (tenant_id, connection_id, tokens) in rows {
        let secret_id = uuid::Uuid::now_v7().to_string();
        conn.execute(
            "INSERT INTO secrets (tenant_id, id, data, key_version, created_at, updated_at)
             VALUES (?1, ?2, ?3, NULL, ?4, ?4)",
            rusqlite::params![tenant_id, secret_id, tokens, now],
        )?;
        conn.execute(
            "INSERT INTO connector_credentials_v2
                 (tenant_id, connection_id, subject, secret_id)
             VALUES (?1, ?2, '', ?3)",
            rusqlite::params![tenant_id, connection_id, secret_id],
        )?;
    }

    conn.execute_batch(
        "DROP TABLE connector_credentials;
         ALTER TABLE connector_credentials_v2 RENAME TO connector_credentials;",
    )
}
