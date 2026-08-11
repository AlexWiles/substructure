//! Reads a session: the list, one session, and its events.
//!
//! A read needs the two stores and nothing else, so a caller with no engine can
//! use it. `Runtime` delegates here.

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

    /// Only a frontend caller is answerable to an owner.
    pub async fn authorize(&self, session_id: &str, caller: &Caller) -> Result<(), RuntimeError> {
        let Caller::Frontend { tenant_id, .. } = caller else {
            return Ok(());
        };

        let session = match self.store.load(tenant_id, session_id).await {
            Ok(session) => session,
            // An uncreated session has no owner yet — nothing to leak; the read
            // is simply empty, and the first turn binds the session to its owner.
            Err(StoreError::StreamNotFound) => return Ok(()),
            Err(e) => return Err(internal(e)),
        };

        match session.state.owner.as_ref().is_some_and(|o| caller.owns(o)) {
            true => Ok(()),
            false => Err(RuntimeError::Session(SessionError::SessionAccessDenied)),
        }
    }
}

fn internal(e: StoreError) -> RuntimeError {
    RuntimeError::Internal(e.to_string())
}
