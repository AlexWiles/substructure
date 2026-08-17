# Database down

1. Check the connection pool first: `pool_active` at zero with traffic means
   the database, not the app.
2. Restart the replica, never the primary: `db-ctl restart replica-1`.
3. Watch replication lag until it is under one second.
4. If the primary itself is down, do nothing else — that is an escalation.
