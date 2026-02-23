use std::sync::Arc;

use ractor::{Actor, ActorProcessingErr, ActorRef};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::domain::event::Event;
use crate::runtime::session_client::ClientMessage;

use super::translate::{EventTranslator, TranslateOutput};
use super::types::AgUiEvent;

// ---------------------------------------------------------------------------
// AgUiObserverActor — joins the session process group, bridges to mpsc
// ---------------------------------------------------------------------------

pub struct AgUiObserverActor;

pub struct ObserverState {
    translator: EventTranslator,
    event_tx: mpsc::Sender<AgUiEvent>,
    thread_id: String,
    run_id: String,
    skip_until: u64,
}

pub struct ObserverArgs {
    pub session_id: Uuid,
    pub thread_id: String,
    pub run_id: String,
    pub event_tx: mpsc::Sender<AgUiEvent>,
    pub init_events: Vec<AgUiEvent>,
    /// Events with sequence <= this value are skipped (0 = skip nothing).
    pub skip_until: u64,
}

impl Actor for AgUiObserverActor {
    type Msg = ClientMessage;
    type State = ObserverState;
    type Arguments = ObserverArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        // Join the session-clients process group so the dispatcher fans events to us
        let group = format!("session-clients-{}", args.session_id);
        ractor::pg::join(group, vec![myself.get_cell()]);

        // Send initial events (RunStarted or MessagesSnapshot)
        for event in args.init_events {
            let _ = args.event_tx.send(event).await;
        }

        Ok(ObserverState {
            translator: EventTranslator::new(),
            event_tx: args.event_tx,
            thread_id: args.thread_id,
            run_id: args.run_id,
            skip_until: args.skip_until,
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            ClientMessage::Events(events) => {
                for event in &events {
                    if !should_translate(event, state.skip_until) {
                        continue;
                    }

                    let output = state.translator.translate(event);
                    match output {
                        TranslateOutput::Events(ag_events) => {
                            for e in ag_events {
                                let _ = state.event_tx.send(e).await;
                            }
                        }
                        TranslateOutput::Terminal(ag_events) => {
                            for e in ag_events {
                                let _ = state.event_tx.send(e).await;
                            }
                            let _ = state
                                .event_tx
                                .send(AgUiEvent::RunFinished {
                                    thread_id: state.thread_id.clone(),
                                    run_id: state.run_id.clone(),
                                    outcome: None,
                                    interrupt: None,
                                })
                                .await;
                            myself.stop(None);
                            return Ok(());
                        }
                        TranslateOutput::Interrupt(ag_events, info) => {
                            for e in ag_events {
                                let _ = state.event_tx.send(e).await;
                            }
                            let _ = state
                                .event_tx
                                .send(AgUiEvent::RunFinished {
                                    thread_id: state.thread_id.clone(),
                                    run_id: state.run_id.clone(),
                                    outcome: Some("interrupt".into()),
                                    interrupt: Some(info),
                                })
                                .await;
                            myself.stop(None);
                            return Ok(());
                        }
                    }
                }
            }
            // SendCommand and GetState are never sent by the dispatcher,
            // but the type system requires exhaustive matching.
            ClientMessage::SendCommand(_, _) | ClientMessage::GetState(_) => {}
        }
        Ok(())
    }
}

fn should_translate(event: &Arc<Event>, skip_until: u64) -> bool {
    if skip_until == 0 {
        return true;
    }
    event.sequence > skip_until
}
