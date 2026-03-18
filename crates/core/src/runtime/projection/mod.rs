use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;

use crate::runtime::event_store::{Event, EventFilter, EventStore};

#[derive(Debug, Clone)]
pub struct ProjectionCheckpoint {
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
pub enum ProjectionError {
    #[error("projection apply failed: {0}")]
    Apply(String),
}

#[async_trait]
pub trait Projection: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn shard_key(&self, event: &Event) -> Option<String>;
    async fn apply(&self, event: &Event) -> Result<(), ProjectionError>;
}

#[async_trait]
pub trait ProjectionCheckpointStore: Send + Sync {
    async fn load_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
    ) -> Result<ProjectionCheckpoint, CheckpointError>;

    async fn compare_and_set_checkpoint(
        &self,
        projection: &str,
        shard_id: u32,
        expected_version: u64,
        new_position: u64,
        owner_id: Option<&str>,
    ) -> Result<bool, CheckpointError>;
}

#[derive(Debug, Clone)]
pub struct ProjectionRunnerConfig {
    pub shard_id: u32,
    pub shard_count: u32,
    pub batch_size: usize,
    pub idle_poll_interval: Duration,
    pub error_backoff: Duration,
    pub owner_id: Option<String>,
}

impl Default for ProjectionRunnerConfig {
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

pub struct ProjectionRunner {
    store: Arc<dyn EventStore>,
    checkpoint_store: Arc<dyn ProjectionCheckpointStore>,
    projection: Arc<dyn Projection>,
    config: ProjectionRunnerConfig,
}

impl ProjectionRunner {
    pub fn new(
        store: Arc<dyn EventStore>,
        checkpoint_store: Arc<dyn ProjectionCheckpointStore>,
        projection: Arc<dyn Projection>,
        config: ProjectionRunnerConfig,
    ) -> Self {
        assert!(config.shard_count > 0, "projection shard_count must be > 0");
        assert!(
            config.shard_id < config.shard_count,
            "projection shard_id must be < shard_count"
        );
        Self {
            store,
            checkpoint_store,
            projection,
            config,
        }
    }

    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            self.run().await;
        })
    }

    pub async fn run(self) {
        let projection_name = self.projection.name();
        let mut checkpoint = loop {
            match self
                .checkpoint_store
                .load_checkpoint(projection_name, self.config.shard_id)
                .await
            {
                Ok(cp) => break cp,
                Err(err) => {
                    tracing::error!(
                        projection = projection_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to load projection checkpoint"
                    );
                    tokio::time::sleep(self.config.error_backoff).await;
                }
            }
        };

        let mut wake_rx = self.store.subscribe();

        'run: loop {
            let events = match self
                .store
                .query_events(&EventFilter {
                    after_position: Some(checkpoint.position),
                    limit: Some(self.config.batch_size),
                    ..Default::default()
                })
                .await
            {
                Ok(events) => events,
                Err(err) => {
                    tracing::error!(
                        projection = projection_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to read projection events"
                    );
                    tokio::time::sleep(self.config.error_backoff).await;
                    continue;
                }
            };

            if events.is_empty() {
                tokio::select! {
                    _ = tokio::time::sleep(self.config.idle_poll_interval) => {}
                    _ = wake_rx.recv() => {}
                }
                continue;
            }

            let mut committable_position = checkpoint.position;

            for event in &events {
                if !event_matches_shard(event, self.projection.as_ref(), self.config.shard_count, self.config.shard_id) {
                    committable_position = committable_position.max(event.position);
                    continue;
                }
                if let Err(err) = self.projection.apply(event).await {
                    tracing::error!(
                        projection = projection_name,
                        shard_id = self.config.shard_id,
                        event_position = event.position,
                        error = %err,
                        "projection apply failed"
                    );

                    if committable_position > checkpoint.position {
                        self.commit_checkpoint_position(
                            projection_name,
                            &mut checkpoint,
                            committable_position,
                        )
                        .await;
                    }

                    tokio::time::sleep(self.config.error_backoff).await;
                    continue 'run;
                }

                committable_position = committable_position.max(event.position);
            }

            if committable_position == checkpoint.position {
                continue;
            }

            self.commit_checkpoint_position(projection_name, &mut checkpoint, committable_position)
                .await;
        }
    }

    async fn commit_checkpoint_position(
        &self,
        projection_name: &str,
        checkpoint: &mut ProjectionCheckpoint,
        new_position: u64,
    ) {
        match self
            .checkpoint_store
            .compare_and_set_checkpoint(
                projection_name,
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
                .load_checkpoint(projection_name, self.config.shard_id)
                .await
            {
                Ok(cp) => *checkpoint = cp,
                Err(err) => {
                    tracing::error!(
                        projection = projection_name,
                        shard_id = self.config.shard_id,
                        error = %err,
                        "failed to reload checkpoint after CAS conflict"
                    );
                }
            },
            Err(err) => {
                tracing::error!(
                    projection = projection_name,
                    shard_id = self.config.shard_id,
                    error = %err,
                    "failed to commit projection checkpoint"
                );
            }
        }
    }
}

fn event_matches_shard(
    event: &Event,
    projection: &dyn Projection,
    shard_count: u32,
    shard_id: u32,
) -> bool {
    debug_assert!(shard_count > 0, "projection shard_count must be > 0");
    let key = match projection.shard_key(event) {
        Some(k) => k,
        None => return false,
    };
    (stable_hash(&key) % u64::from(shard_count)) as u32 == shard_id
}

fn stable_hash(input: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
