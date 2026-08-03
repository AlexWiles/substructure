use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::{EventFilter, EventStore, Seq, StoreError};
use crate::runtime::session::SessionEvent;

/// One session's event stream, named.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StreamRef {
    pub tenant_id: String,
    pub session_id: String,
}

/// How far a processor has read one stream. `version` guards the CAS, so two
/// owners racing the same stream cannot both advance it.
#[derive(Debug, Clone)]
pub struct StreamCursor {
    pub seq: Seq,
    pub version: u64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum CursorError {
    #[error("cursor error: {0}")]
    Message(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessorError {
    #[error("processor apply failed: {0}")]
    Apply(String),
}

#[async_trait]
pub trait EventProcessor: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    async fn apply(&self, event: SessionEvent) -> Result<(), ProcessorError>;
}

/// Per-stream read positions for a processor. There is no store-wide cursor:
/// a processor is exactly as far along as its slowest stream, and streams
/// advance independently.
#[async_trait]
pub trait ProcessorCursorStore: Send + Sync {
    /// Where the processor left off. A stream never read comes back at seq 0,
    /// version 0.
    async fn load_cursor(
        &self,
        processor: &str,
        stream: &StreamRef,
    ) -> Result<StreamCursor, CursorError>;

    async fn compare_and_set_cursor(
        &self,
        processor: &str,
        stream: &StreamRef,
        expected_version: u64,
        new_seq: Seq,
        owner_id: Option<&str>,
    ) -> Result<bool, CursorError>;

    /// Streams in this shard holding events past the processor's cursor —
    /// the work list. Implementations own the shard split so an unread
    /// stream is never fetched just to be discarded.
    async fn pending_streams(
        &self,
        processor: &str,
        shard_id: u32,
        shard_count: u32,
        limit: usize,
    ) -> Result<Vec<StreamRef>, CursorError>;

    /// Park every existing stream's cursor at its head, once per processor:
    /// how a new processor renders the future, not the whole history. A
    /// second call is a no-op, and a stream created later still starts at 0.
    async fn seed_at_tail(&self, processor: &str) -> Result<(), CursorError>;
}

#[derive(Debug, Clone)]
pub struct EventProcessorRunnerConfig {
    pub shard_id: u32,
    pub shard_count: u32,
    /// Events read per stream per round trip.
    pub batch_size: usize,
    /// Streams claimed per sweep.
    pub stream_batch: usize,
    pub idle_poll_interval: Duration,
    pub error_backoff: Duration,
    pub owner_id: Option<String>,
}

impl Default for EventProcessorRunnerConfig {
    fn default() -> Self {
        Self {
            shard_id: 0,
            shard_count: 1,
            batch_size: 256,
            stream_batch: 128,
            idle_poll_interval: Duration::from_millis(500),
            error_backoff: Duration::from_secs(1),
            owner_id: None,
        }
    }
}

pub struct EventProcessorRunner {
    store: Arc<dyn EventStore>,
    cursor_store: Arc<dyn ProcessorCursorStore>,
    processor: Arc<dyn EventProcessor>,
    config: EventProcessorRunnerConfig,
    cancel: CancellationToken,
}

/// A stream that made no progress this round. Its cursor stands, so the next
/// sweep retries it; other streams are unaffected.
struct DrainFailed;

impl EventProcessorRunner {
    pub fn new(
        store: Arc<dyn EventStore>,
        cursor_store: Arc<dyn ProcessorCursorStore>,
        processor: Arc<dyn EventProcessor>,
        config: EventProcessorRunnerConfig,
        cancel: CancellationToken,
    ) -> Self {
        assert!(config.shard_count > 0, "processor shard_count must be > 0");
        assert!(
            config.shard_id < config.shard_count,
            "processor shard_id must be < shard_count"
        );
        Self {
            store,
            cursor_store,
            processor,
            config,
            cancel,
        }
    }

    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            self.run().await;
        })
    }

    pub async fn run(self) {
        let name = self.processor.name();
        let mut wake_rx = self.store.subscribe();

        loop {
            if self.cancel.is_cancelled() {
                break;
            }

            let streams = match self
                .cursor_store
                .pending_streams(
                    name,
                    self.config.shard_id,
                    self.config.shard_count,
                    self.config.stream_batch,
                )
                .await
            {
                Ok(streams) => streams,
                Err(_) if self.cancel.is_cancelled() => break,
                Err(err) => {
                    tracing::error!(
                        processor = name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to list pending streams"
                    );
                    tokio::time::sleep(self.config.error_backoff).await;
                    continue;
                }
            };

            if streams.is_empty() {
                tokio::select! {
                    _ = tokio::time::sleep(self.config.idle_poll_interval) => {}
                    _ = wake_rx.recv() => {}
                    _ = self.cancel.cancelled() => break,
                }
                continue;
            }

            // A stuck stream backs itself off without holding up the rest.
            let mut backoff = false;
            for stream in &streams {
                if self.cancel.is_cancelled() {
                    break;
                }
                if self.drain(name, stream).await.is_err() {
                    backoff = true;
                }
            }
            if backoff {
                tokio::time::sleep(self.config.error_backoff).await;
            }
        }
    }

    /// Apply one stream's unread events in `seq` order, committing the cursor
    /// per batch. Stops at the first apply failure, keeping what came before.
    async fn drain(&self, name: &str, stream: &StreamRef) -> Result<(), DrainFailed> {
        let mut cursor = match self.cursor_store.load_cursor(name, stream).await {
            Ok(cursor) => cursor,
            Err(err) => {
                tracing::error!(
                    processor = name,
                    session_id = %stream.session_id,
                    error = %err,
                    "failed to load stream cursor"
                );
                return Err(DrainFailed);
            }
        };

        loop {
            if self.cancel.is_cancelled() {
                return Ok(());
            }

            let events = match self
                .store
                .query_events(&EventFilter {
                    tenant_id: Some(stream.tenant_id.clone()),
                    session_id: Some(stream.session_id.clone()),
                    after_seq: Some(cursor.seq),
                    limit: Some(self.config.batch_size),
                })
                .await
            {
                Ok(events) => events,
                Err(StoreError::Cancelled) => return Ok(()),
                Err(err) => {
                    tracing::error!(
                        processor = name,
                        session_id = %stream.session_id,
                        error = %err,
                        "failed to read stream events"
                    );
                    return Err(DrainFailed);
                }
            };

            if events.is_empty() {
                return Ok(());
            }
            let read = events.len();

            let mut applied = cursor.seq;
            let mut failure = None;
            for event in events {
                let seq = Seq(event.seq);
                if let Err(err) = self.processor.apply(event).await {
                    failure = Some((seq, err));
                    break;
                }
                applied = seq;
            }

            if applied > cursor.seq && !self.commit(name, stream, &mut cursor, applied).await {
                // Another owner holds this stream; leave it to them.
                return Ok(());
            }

            if let Some((seq, err)) = failure {
                tracing::error!(
                    processor = name,
                    session_id = %stream.session_id,
                    seq = seq.0,
                    error = %err,
                    "processor apply failed"
                );
                return Err(DrainFailed);
            }

            if read < self.config.batch_size {
                return Ok(());
            }
        }
    }

    /// Advance the stream cursor. `false` means the CAS was lost — the cursor
    /// is reloaded, but this runner stops touching the stream this round.
    async fn commit(
        &self,
        name: &str,
        stream: &StreamRef,
        cursor: &mut StreamCursor,
        new_seq: Seq,
    ) -> bool {
        match self
            .cursor_store
            .compare_and_set_cursor(
                name,
                stream,
                cursor.version,
                new_seq,
                self.config.owner_id.as_deref(),
            )
            .await
        {
            Ok(true) => {
                cursor.seq = new_seq;
                cursor.version += 1;
                cursor.updated_at = Utc::now();
                true
            }
            Ok(false) => false,
            Err(err) => {
                tracing::error!(
                    processor = name,
                    session_id = %stream.session_id,
                    error = %err,
                    "failed to commit stream cursor"
                );
                false
            }
        }
    }
}
