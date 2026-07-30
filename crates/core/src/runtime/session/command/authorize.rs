//! Who may do what. Every command passes through here before it writes.
//!
//! Three callers, in descending privilege: [`Caller::System`] is the engine
//! itself, [`Caller::Machine`] a worker, [`Caller::Frontend`] an end user. A
//! check that names a caller lives here; a check that names an effect's own
//! handler is the kind's, in [`effects`](super::super::effects).

use super::SessionError;
use crate::protocol::InterruptOrigin;
use crate::runtime::session::decision::{LlmHandler, ToolHandler};
use crate::runtime::session::state::{LlmCallState, SessionState, ToolCallState};
use crate::runtime::Caller;

impl SessionState {
    pub(in crate::runtime::session) fn ensure_internal(
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            _ => Err(SessionError::SessionAccessDenied),
        }
    }

    pub(in crate::runtime::session) fn ensure_machine_or_system(
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::SessionAccessDenied),
        }
    }

    pub(super) fn ensure_tenant_matches(
        caller: &Caller,
        tenant_id: &str,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Machine {
                tenant_id: caller_tenant,
                ..
            }
            | Caller::Frontend {
                tenant_id: caller_tenant,
                ..
            } => {
                if caller_tenant != tenant_id {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    pub(super) fn caller_interrupt_origin(caller: &Caller) -> InterruptOrigin {
        match caller {
            Caller::System { .. } => InterruptOrigin::System,
            Caller::Machine { .. } => InterruptOrigin::Machine,
            Caller::Frontend { .. } => InterruptOrigin::Frontend,
        }
    }

    pub(super) fn ensure_owns_session(&self, caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::Machine { .. } => Ok(()),
            Caller::Frontend { user_id, .. } => {
                let owner = self
                    .owner
                    .as_ref()
                    .ok_or(SessionError::SessionAccessDenied)?;
                if owner.id.as_deref() != Some(user_id.as_str()) {
                    return Err(SessionError::SessionAccessDenied);
                }
                Ok(())
            }
        }
    }

    /// A frontend may answer only the calls it was asked to run.
    pub(in crate::runtime::session) fn check_tool_call_caller(
        &self,
        tc: &ToolCallState,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        self.ensure_owns_session(caller)?;
        if matches!(caller, Caller::Frontend { .. }) && tc.handler != ToolHandler::Client {
            return Err(SessionError::EffectWrongHandler);
        }
        Ok(())
    }

    /// A worker may answer only the calls it was handed; a frontend, none.
    pub(in crate::runtime::session) fn check_llm_call_caller(
        &self,
        call: Option<&LlmCallState>,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::EffectWrongHandler),
            Caller::Machine { .. } => match call {
                Some(c) if c.handler == LlmHandler::Worker => Ok(()),
                Some(_) => Err(SessionError::EffectWrongHandler),
                None => Err(SessionError::EffectNotFound),
            },
        }
    }
}
