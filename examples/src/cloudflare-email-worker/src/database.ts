import { DurableObject } from "cloudflare:workers";

export interface EmailRow {
    id: number;
    message_id: string | null;
    thread_id: string | null;
    in_reply_to: string | null;
    from_addr: string;
    to_addr: string;
    subject: string;
    body: string;
    received_at: number;
}

export interface ExpenseRow {
    id: number;
    email_id: number;
    amount_cents: number;
    description: string;
    created_at: number;
}

export interface RecordEmailInput {
    messageId: string | null;
    threadId: string | null;
    inReplyTo: string | null;
    from: string;
    to: string;
    subject: string;
    body: string;
}

export interface RecordExpenseInput {
    emailId: number;
    amountCents: number;
    description: string;
}

export class Database extends DurableObject<Env> {
    constructor(ctx: DurableObjectState, env: Env) {
        super(ctx, env);
        ctx.blockConcurrencyWhile(async () => {
            ctx.storage.sql.exec(`
                CREATE TABLE IF NOT EXISTS emails (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id  TEXT    UNIQUE,
                    thread_id   TEXT,
                    in_reply_to TEXT,
                    from_addr   TEXT    NOT NULL,
                    to_addr     TEXT    NOT NULL,
                    subject     TEXT    NOT NULL,
                    body        TEXT    NOT NULL,
                    received_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
                )
            `);
            ctx.storage.sql.exec("CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_id)");
            ctx.storage.sql.exec(`
                CREATE TABLE IF NOT EXISTS expenses (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id     INTEGER NOT NULL REFERENCES emails(id),
                    amount_cents INTEGER NOT NULL,
                    description  TEXT    NOT NULL,
                    created_at   INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
                )
            `);
            ctx.storage.sql.exec("CREATE INDEX IF NOT EXISTS idx_expenses_email ON expenses(email_id)");
        });
    }

    // Upsert on message_id so duplicate deliveries reuse the existing row.
    async recordEmail(input: RecordEmailInput): Promise<number> {
        const row = this.ctx.storage.sql
            .exec<{ id: number }>(
                `INSERT INTO emails
                   (message_id, thread_id, in_reply_to, from_addr, to_addr, subject, body)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(message_id) DO UPDATE SET message_id = excluded.message_id
                 RETURNING id`,
                input.messageId,
                input.threadId,
                input.inReplyTo,
                input.from,
                input.to,
                input.subject,
                input.body,
            )
            .one();
        return row.id;
    }

    async recordExpense(input: RecordExpenseInput): Promise<void> {
        this.ctx.storage.sql.exec(
            `INSERT INTO expenses (email_id, amount_cents, description)
             VALUES (?, ?, ?)`,
            input.emailId,
            input.amountCents,
            input.description,
        );
    }

    async query(sql: string, params: SqlStorageValue[] = []): Promise<Record<string, SqlStorageValue>[]> {
        return this.ctx.storage.sql.exec(sql, ...params).toArray();
    }

    async getEmail(id: number): Promise<EmailRow | null> {
        const rows = this.ctx.storage.sql.exec("SELECT * FROM emails WHERE id = ?", id).toArray();
        return rows[0] ? (rows[0] as unknown as EmailRow) : null;
    }
}
