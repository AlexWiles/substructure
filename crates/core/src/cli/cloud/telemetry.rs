use std::sync::OnceLock;

use uuid::Uuid;

#[derive(Debug)]
pub struct Telemetry {
    pub invocation_id: String,
    pub command: &'static str,
}

static STATE: OnceLock<Telemetry> = OnceLock::new();

pub fn init(command: &'static str) {
    let _ = STATE.set(Telemetry {
        invocation_id: Uuid::now_v7().to_string(),
        command,
    });
}

pub fn get() -> Option<&'static Telemetry> {
    STATE.get()
}
