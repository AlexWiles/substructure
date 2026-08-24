//! Starting one turn and subscribing to it.
//!
//! Shared by every caller that submits one input and streams what it opens:
//! `subs run` and `subs chat` on an engine here, and the
//! `/api/v1/projects/{project}/run` route a deployment serves.

use tokio::sync::mpsc;

use crate::event_store::Seq;
use crate::protocol::{ClientInput, SessionOwner, TokenDelta};
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::session::SessionEvent;
use crate::span::SpanContext;
use crate::transport::channel::ChannelContext;
use crate::{Caller, HandleClientInput, RuntimeError};

/// A turn that has started, and the streams that carry it.
pub struct Turn {
    pub turn_id: String,
    pub events: mpsc::Receiver<SessionEvent>,
    /// Only a translated reader subscribes: a token delta is a fragment of an
    /// AG-UI message, and means nothing to a reader of engine events.
    pub deltas: Option<mpsc::Receiver<TokenDelta>>,
}

/// Submit `input` and open the streams for the turn it starts.
///
/// The event stream is scoped to that turn, so it carries nothing from a turn
/// running beside it and closes itself when the turn completes. `base_seq` is
/// read before the submit and replayed after it, which is what lets the
/// subscribe follow the submit without losing the events in between.
pub async fn start(
    ctx: &ChannelContext,
    caller: &Caller,
    owner: &SessionOwner,
    session_id: &str,
    input: ClientInput,
    span: &str,
    translated: bool,
) -> Result<Turn, RuntimeError> {
    let deltas = match translated {
        true => Some(ctx.subscribe_token_deltas(caller, session_id).await),
        false => None,
    };

    let base_seq = match ctx.get_session(caller.tenant_id(), session_id).await {
        Ok(session) => Seq(session.seq),
        Err(_) => Seq(0),
    };

    let turn_id = ctx
        .handle_client_input(HandleClientInput {
            session_id: session_id.to_string(),
            caller: caller.clone(),
            owner: owner.clone(),
            input,
            span: SpanContext::root().child(span),
        })
        .await?
        .turn_id;

    let events = ctx
        .stream(
            SessionSubscriptionSpec {
                session_id: session_id.to_string(),
                caller: caller.clone(),
                scope: SubscriptionScope::Turn {
                    turn_id: turn_id.clone(),
                },
            },
            Some(base_seq),
        )
        .await?;

    Ok(Turn {
        turn_id,
        events,
        deltas,
    })
}
