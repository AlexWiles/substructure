use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::runtime::event_store::{EventFilter, EventStore, GlobalPosition};
use crate::runtime::session::SessionEvent;
use crate::shard::in_shard;

#[derive(Debug, Clone)]
pub struct ProcessorCheckpoint {
    pub position: u64,
    pub version: u64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint error: {0}")]
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

#[async_trait]
pub trait ProcessorCheckpointStore: Send + Sync {
    async fn load_checkpoint(
        &self,
        processor: &str,
        shard_id: u32,
    ) -> Result<ProcessorCheckpoint, CheckpointError>;

    async fn compare_and_set_checkpoint(
        &self,
        processor: &str,
        shard_id: u32,
        expected_version: u64,
        new_position: u64,
        owner_id: Option<&str>,
    ) -> Result<bool, CheckpointError>;
}

#[derive(Debug, Clone)]
pub struct EventProcessorRunnerConfig {
    pub shard_id: u32,
    pub shard_count: u32,
    pub batch_size: usize,
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
            idle_poll_interval: Duration::from_millis(500),
            error_backoff: Duration::from_secs(1),
            owner_id: None,
        }
    }
}

pub struct EventProcessorRunner {
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
    processor: Arc<dyn EventProcessor>,
    config: EventProcessorRunnerConfig,
    cancel: CancellationToken,
}

enum BatchProcessError {
    ApplyFailed,
}

impl EventProcessorRunner {
    pub fn new(
        store: Arc<dyn EventStore>,
        checkpoint_store: Arc<dyn ProcessorCheckpointStore>,
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
            checkpoint_store,
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
        let processor_name = self.processor.name();
        let mut checkpoint = loop {
            match self
                .checkpoint_store
                .load_checkpoint(processor_name, self.config.shard_id)
                .await
            {
                Ok(cp) => break cp,
                Err(err) => {
                    tracing::error!(
                        processor = processor_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to load processor checkpoint"
                    );
                    tokio::time::sleep(self.config.error_backoff).await;
                }
            }
        };

        let mut wake_rx = self.store.subscribe();

        loop {
            if self.cancel.is_cancelled() {
                break;
            }

            let events = match self
                .store
                .query_events(&EventFilter {
                    after_global_position: Some(GlobalPosition(checkpoint.position)),
                    limit: Some(self.config.batch_size),
                    ..Default::default()
                })
                .await
            {
                Ok(events) => events,
                Err(_) if self.cancel.is_cancelled() => break,
                Err(err) => {
                    tracing::error!(
                        processor = processor_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to read processor events"
                    );
                    tokio::time::sleep(self.config.error_backoff).await;
                    continue;
                }
            };

            if events.is_empty() {
                tokio::select! {
                    _ = tokio::time::sleep(self.config.idle_poll_interval) => {}
                    _ = wake_rx.recv() => {}
                    _ = self.cancel.cancelled() => break,
                }
                continue;
            }

            let _ = self
                .process_batch(processor_name, &mut checkpoint, &events)
                .await;
        }
    }

    async fn process_batch(
        &self,
        processor_name: &str,
        checkpoint: &mut ProcessorCheckpoint,
        events: &[SessionEvent],
    ) -> Result<(), BatchProcessError> {
        let mut committable_position = checkpoint.position;

        for event in events {
            if in_shard(
                &event.session_id,
                self.config.shard_count,
                self.config.shard_id,
            ) {
                if let Err(err) = self.processor.apply(event.clone()).await {
                    self.record_failure(
                        processor_name,
                        checkpoint,
                        committable_position,
                        event.global_position.0,
                        err.to_string(),
                    )
                    .await;
                    return Err(BatchProcessError::ApplyFailed);
                }
            }

            committable_position = committable_position.max(event.global_position.0);
        }

        if committable_position == checkpoint.position {
            return Ok(());
        }

        self.commit_checkpoint_position(processor_name, checkpoint, committable_position)
            .await;
        Ok(())
    }

    async fn commit_checkpoint_position(
        &self,
        processor_name: &str,
        checkpoint: &mut ProcessorCheckpoint,
        new_position: u64,
    ) {
        match self
            .checkpoint_store
            .compare_and_set_checkpoint(
                processor_name,
                self.config.shard_id,
                checkpoint.version,
                new_position,
                self.config.owner_id.as_deref(),
            )
            .await
        {
            Ok(true) => {
                checkpoint.position = new_position;
                checkpoint.version += 1;
                checkpoint.updated_at = Utc::now();
            }
            Ok(false) => match self
                .checkpoint_store
                .load_checkpoint(processor_name, self.config.shard_id)
                .await
            {
                Ok(cp) => *checkpoint = cp,
                Err(err) => {
                    tracing::error!(
                        processor = processor_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to reload checkpoint after CAS conflict"
                    );
                }
            },
            Err(err) => {
                tracing::error!(
                    processor = processor_name,
                    shard_id = self.config.shard_id,
                    error = %err,
                    "failed to commit processor checkpoint"
                );

                match self
                    .checkpoint_store
                    .load_checkpoint(processor_name, self.config.shard_id)
                    .await
                {
                    Ok(cp) => *checkpoint = cp,
                    Err(load_err) => {
                        tracing::error!(
                            processor = processor_name,
                            shard_id = self.config.shard_id,
                            error = %load_err,
                            "failed to reload checkpoint after commit error"
                        );
                    }
                }

                tokio::time::sleep(self.config.error_backoff).await;
            }
        }
    }

    async fn record_failure(
        &self,
        processor_name: &str,
        checkpoint: &mut ProcessorCheckpoint,
        committable_position: u64,
        event_position: u64,
        error: String,
    ) {
        tracing::error!(
            processor = processor_name,
            shard_id = self.config.shard_id,
            event_position,
            error = %error,
            "processor apply failed"
        );
        if committable_position > checkpoint.position {
            self.commit_checkpoint_position(processor_name, checkpoint, committable_position)
                .await;
        }
        tokio::time::sleep(self.config.error_backoff).await;
    }
}
