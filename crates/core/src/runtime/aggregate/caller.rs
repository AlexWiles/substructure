use std::collections::HashMap;

/// Identifies who is invoking a command on an aggregate.
///
/// Threaded through [`ExecuteInput`](super::ExecuteInput) so command
/// handlers can enforce authorization at the domain layer rather than
/// the transport edge.
#[derive(Debug, Clone)]
pub enum Caller {
    /// Internal/system call. Used by background processors, wake cycles,
    /// and recursive expansions inside a command. No authorization is
    /// enforced.
    System,
    /// External call authenticated as a machine principal (API key).
    /// Trusted to act on any aggregate within its tenant. `key_id`
    /// identifies the credential that authenticated — not a person.
    Machine { tenant_id: String, key_id: String },
    /// External call authenticated as an end user via a frontend token
    /// (JWT). Scoped to aggregates the user owns; command handlers verify
    /// by comparing `user_id` against the owner recorded on the aggregate.
    /// `attrs` carries arbitrary claims the worker minted into the token
    /// (e.g. role, plan tier) for use in domain-level policy.
    Frontend {
        tenant_id: String,
        user_id: String,
        attrs: HashMap<String, String>,
    },
}
