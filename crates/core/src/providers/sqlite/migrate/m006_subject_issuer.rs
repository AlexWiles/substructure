//! A slot is keyed by a source and a name, not a name alone: one person called
//! `bob` by an application is not another called `bob` by a workspace.
//!
//! Rows written before this are the deployment's own credential, which no
//! source names, and they carry through. A personal row from the one release
//! that had them is dropped with its secret: no issuer can be inferred for it,
//! and guessing would hand a credential to the wrong person.

use rusqlite::Connection;

pub fn up(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "DELETE FROM secrets WHERE (tenant_id, id) IN (
             SELECT tenant_id, secret_id FROM connector_credentials WHERE subject <> ''
         );
         CREATE TABLE connector_credentials_v3 (
            tenant_id     TEXT NOT NULL,
            connection_id TEXT NOT NULL,
            issuer        TEXT NOT NULL,
            subject       TEXT NOT NULL,
            secret_id     TEXT NOT NULL,
            PRIMARY KEY (tenant_id, connection_id, issuer, subject)
         );
         INSERT INTO connector_credentials_v3
                (tenant_id, connection_id, issuer, subject, secret_id)
              SELECT tenant_id, connection_id, '', '', secret_id
                FROM connector_credentials
               WHERE subject = '';
         DROP TABLE connector_credentials;
         ALTER TABLE connector_credentials_v3 RENAME TO connector_credentials;",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::sqlite::migrate::{m001_baseline, m004_secret_store};

    /// The deployment's own credential survives; a personal one from before
    /// issuers existed goes, secret and all.
    #[test]
    fn the_shared_slot_carries_through_and_a_sourceless_person_does_not() {
        let conn = Connection::open_in_memory().unwrap();
        m001_baseline::up(&conn).unwrap();
        m004_secret_store::up(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO secrets (tenant_id, id, data, created_at, updated_at)
                  VALUES ('t', 'sec-shared', 'x', '', ''), ('t', 'sec-person', 'y', '', '');
             INSERT INTO connector_credentials (tenant_id, connection_id, subject, secret_id)
                  VALUES ('t', 'gmail', '', 'sec-shared'), ('t', 'gmail', 'U1', 'sec-person');",
        )
        .unwrap();

        up(&conn).unwrap();

        let held: Vec<(String, String, String)> = {
            let mut stmt = conn
                .prepare("SELECT issuer, subject, secret_id FROM connector_credentials")
                .unwrap();
            stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        assert_eq!(held, [(String::new(), String::new(), "sec-shared".into())]);

        let secrets: Vec<String> = {
            let mut stmt = conn.prepare("SELECT id FROM secrets ORDER BY id").unwrap();
            stmt.query_map([], |row| row.get(0))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        assert_eq!(secrets, ["sec-shared"], "the orphaned secret goes too");
    }
}
