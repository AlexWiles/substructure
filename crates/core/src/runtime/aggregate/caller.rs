use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Caller {
    System {
        tenant_id: String,
    },
    Machine {
        tenant_id: String,
        key_id: String,
    },
    Frontend {
        tenant_id: String,
        user_id: String,
        attrs: HashMap<String, String>,
    },
}

impl Caller {
    pub fn tenant_id(&self) -> &str {
        match self {
            Caller::System { tenant_id }
            | Caller::Machine { tenant_id, .. }
            | Caller::Frontend { tenant_id, .. } => tenant_id,
        }
    }
}
