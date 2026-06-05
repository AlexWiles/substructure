use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Caller {
    /// Internal/system call (background processors, wake cycles, recursive
    /// expansions). Still scoped to the tenant it acts within.
    System { tenant_id: String },
    /// External call authenticated as a machine principal (API key).
    Machine {
        tenant_id: String,
        /// api key id
        key_id: String,
    },
    /// External call authenticated as an end user
    Frontend {
        tenant_id: String,
        user_id: String,
        attrs: HashMap<String, String>,
    },
}

impl Caller {
    /// The tenant this caller acts within. Every caller is tenant-scoped, so the
    /// tenant lives here rather than being threaded alongside the caller.
    pub fn tenant_id(&self) -> &str {
        match self {
            Caller::System { tenant_id }
            | Caller::Machine { tenant_id, .. }
            | Caller::Frontend { tenant_id, .. } => tenant_id,
        }
    }
}
