//! Reading sessions back: the list, one session, and its events.
//!
//! Held apart from [`Runtime`](crate::Runtime) because a read needs two stores
//! and nothing else — no decision loops, no executors, no queue. That is what
//! lets `subs sessions` read a database on this machine the same way the API
//! reads a deployment's, without starting an engine to answer a question about
//! what already happened.
//!
//! `Runtime`'s inspection methods delegate here, so there is one definition of
//! what reading a session means rather than one per caller.

use std::sync::Arc;

use crate::runtime::event_store::{EventFilter, EventStore, Seq, StoreError};
use crate::runtime::session::command::SessionError;
use crate::runtime::session::index::{SessionFilter, SessionIndexStore, SessionPage};
use crate::runtime::session::{SessionAggregate, SessionEvent};
use crate::runtime::RuntimeError;
use crate::Caller;

pub struct SessionReader {
    store: Arc<dyn EventStore>,
    session_index: Arc<dyn SessionIndexStore>,
}

impl SessionReader {
    pub fn new(store: Arc<dyn EventStore>, session_index: Arc<dyn SessionIndexStore>) -> Self {
        Self {
            store,
            session_index,
        }
    }

    pub async fn list(&self, filter: &SessionFilter) -> Result<SessionPage, RuntimeError> {
        self.session_index
            .list_sessions(filter)
            .await
            .map_err(internal)
    }

    pub async fn count(&self, filter: &SessionFilter) -> Result<u64, RuntimeError> {
        self.session_index
            .count_sessions(filter)
            .await
            .map_err(internal)
    }

    pub async fn session(
        &self,
        tenant_id: &str,
        session_id: &str,
    ) -> Result<SessionAggregate, RuntimeError> {
        self.store
            .load(tenant_id, session_id)
            .await
            .map_err(internal)
    }

    pub async fn events(
        &self,
        caller: &Caller,
        session_id: &str,
        after: Option<Seq>,
        limit: Option<usize>,
    ) -> Result<Vec<SessionEvent>, RuntimeError> {
        self.authorize(session_id, caller).await?;
        let filter = EventFilter {
            session_id: Some(session_id.to_string()),
            tenant_id: Some(caller.tenant_id().to_string()),
            after_seq: after,
            limit,
        };
        self.store.query_events(&filter).await.map_err(internal)
    }

    /// Whether this caller may read this session. Only a frontend caller is
    /// answerable to an owner; a machine or the engine itself reads what it
    /// asks for.
    pub async fn authorize(&self, session_id: &str, caller: &Caller) -> Result<(), RuntimeError> {
        let Caller::Frontend {
            tenant_id, user_id, ..
        } = caller
        else {
            return Ok(());
        };

        let session = match self.store.load(tenant_id, session_id).await {
            Ok(session) => session,
            // An uncreated session has no owner yet — nothing to leak; the read
            // is simply empty, and the first turn binds the session to its owner.
            Err(StoreError::StreamNotFound) => return Ok(()),
            Err(e) => return Err(internal(e)),
        };

        let owner_id = session.state.owner.as_ref().and_then(|o| o.id.as_deref());

        if owner_id == Some(user_id.as_str()) {
            Ok(())
        } else {
            Err(RuntimeError::Session(SessionError::SessionAccessDenied))
        }
    }
}

fn internal(e: StoreError) -> RuntimeError {
    RuntimeError::Internal(e.to_string())
}
