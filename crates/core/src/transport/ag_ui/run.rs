use tokio::sync::mpsc;

use crate::event_store::Seq;
use crate::protocol::TokenDelta;
use crate::session::subscriptions::{SessionSubscriptionSpec, SubscriptionScope};
use crate::session::SessionEvent;
use crate::transport::channel::ChannelContext;
use crate::{HandleClientInput, RuntimeError};

pub struct Turn {
    pub turn_id: String,
    pub events: mpsc::Receiver<SessionEvent>,
    pub deltas: Option<mpsc::Receiver<TokenDelta>>,
}

pub async fn start(
    ctx: &ChannelContext,
    input: HandleClientInput,
    translated: bool,
) -> Result<Turn, RuntimeError> {
    let caller = input.caller.clone();
    let session_id = input.session_id.clone();
    let deltas = match translated {
        true => Some(ctx.subscribe_token_deltas(&caller, &session_id).await),
        false => None,
    };

    let base_seq = match ctx.get_session(caller.tenant_id(), &session_id).await {
        Ok(session) => Seq(session.seq),
        Err(_) => Seq(0),
    };

    let turn_id = ctx.handle_client_input(input).await?.turn_id;

    let events = ctx
        .stream(
            SessionSubscriptionSpec {
                session_id,
                caller,
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
