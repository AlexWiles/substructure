//! Who may do what. Every command passes through here before it writes.
//!
//! A check that names a caller lives here. A check that names an effect's own
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

    /// The engine, or a worker answering what it was handed. An admin is not a
    /// worker: no engine gave it a decision to submit.
    pub(in crate::runtime::session) fn ensure_worker_or_system(
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::ApiKey { .. } => Ok(()),
            Caller::Admin { .. } | Caller::Frontend { .. } => {
                Err(SessionError::SessionAccessDenied)
            }
        }
    }

    /// Anyone but an end user. For operations on a session rather than in it.
    pub(in crate::runtime::session) fn ensure_operator_or_system(
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::ApiKey { .. } | Caller::Admin { .. } => Ok(()),
            Caller::Frontend { .. } => Err(SessionError::SessionAccessDenied),
        }
    }

    pub(super) fn ensure_tenant_matches(
        caller: &Caller,
        tenant_id: &str,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::ApiKey {
                tenant_id: caller_tenant,
                ..
            }
            | Caller::Admin {
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
            Caller::Admin { .. } => InterruptOrigin::Admin,
            Caller::ApiKey { .. } => InterruptOrigin::Machine,
            Caller::Frontend { .. } => InterruptOrigin::Frontend,
        }
    }

    pub(super) fn ensure_owns_session(&self, caller: &Caller) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } | Caller::ApiKey { .. } | Caller::Admin { .. } => Ok(()),
            Caller::Frontend { .. } => {
                let owner = self
                    .owner
                    .as_ref()
                    .ok_or(SessionError::SessionAccessDenied)?;
                match caller.owns(owner) {
                    true => Ok(()),
                    false => Err(SessionError::SessionAccessDenied),
                }
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

    /// A worker may answer only the calls it was handed. Nobody else answers
    /// one: an admin administers the session, it does not run the model.
    pub(in crate::runtime::session) fn check_llm_call_caller(
        &self,
        call: Option<&LlmCallState>,
        caller: &Caller,
    ) -> Result<(), SessionError> {
        match caller {
            Caller::System { .. } => Ok(()),
            Caller::Admin { .. } | Caller::Frontend { .. } => Err(SessionError::EffectWrongHandler),
            Caller::ApiKey { .. } => match call {
                Some(c) if c.handler == LlmHandler::Worker => Ok(()),
                Some(_) => Err(SessionError::EffectWrongHandler),
                None => Err(SessionError::EffectNotFound),
            },
        }
    }
}
