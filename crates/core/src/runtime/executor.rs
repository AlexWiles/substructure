use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::protocol::{ErrorCode, ErrorInfo};
use crate::providers::memory_queue::TaskQueue;
use crate::runtime::event_store::EventStore;
use crate::runtime::session::command::{CommandPayload, SettleError};
use crate::runtime::session::effects::{QUEUED, RUN};
use crate::runtime::session::state::EffectKind;
use crate::runtime::session::{execute, ConflictRetry, ExecuteInput};
use crate::runtime::span::SpanContext;
use crate::runtime::Caller;

#[derive(Debug, Clone)]
pub struct TaskBound {
    pub tenant_id: String,
    pub session_id: String,
    pub kind: EffectKind,
    pub id: String,
    pub attempt: Option<u32>,
    pub enqueued_at: DateTime<Utc>,
    pub queue_timeout: Option<Duration>,
    pub run_timeout: Option<Duration>,
    pub span: SpanContext,
}

impl TaskBound {
    fn waited(&self, now: DateTime<Utc>) -> Duration {
        (now - self.enqueued_at).to_std().unwrap_or(Duration::ZERO)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutorPool {
    pub workers: usize,
    pub concurrency: usize,
}

pub trait BoundedTask: Send + 'static {
    fn bound(&self) -> Option<TaskBound>;
}

pub fn spawn_bounded_executors<T, F, Fut>(
    store: Arc<dyn EventStore>,
    queue: Arc<dyn TaskQueue<T>>,
    pool: ExecutorPool,
    cancel: CancellationToken,
    handle: F,
) -> Vec<JoinHandle<()>>
where
    T: BoundedTask,
    F: Fn(T) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send,
{
    let workers = pool.workers.max(1);
    let concurrency = pool.concurrency.max(1);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let store = store.clone();
        let cancel = cancel.clone();
        let handle = handle.clone();
        let rx = queue.subscribe();
        handles.push(tokio::spawn(async move {
            UnboundedReceiverStream::new(rx)
                .take_until(cancel.cancelled_owned())
                .for_each_concurrent(concurrency, |task| {
                    let store = store.clone();
                    let handle = handle.clone();
                    async move {
                        run_bounded(task, handle, |bound, message| {
                            let store = store.clone();
                            async move { expire(store.as_ref(), &bound, message).await }
                        })
                        .await
                    }
                })
                .await;
        }));
    }
    handles
}

async fn run_bounded<T, F, Fut, E, EFut>(task: T, handle: F, expire: E)
where
    T: BoundedTask,
    F: Fn(T) -> Fut,
    Fut: Future<Output = ()>,
    E: Fn(TaskBound, &'static str) -> EFut,
    EFut: Future<Output = ()>,
{
    let Some(bound) = task.bound() else {
        return handle(task).await;
    };
    let waited = bound.waited(Utc::now());
    if bound.queue_timeout.is_some_and(|q| waited > q) {
        tracing::warn!(
            session_id = %bound.session_id,
            kind = bound.kind.label(),
            id = %bound.id,
            waited_secs = waited.as_secs(),
            "dropping a task that waited too long for an executor"
        );
        return expire(bound, QUEUED).await;
    }
    let Some(limit) = bound.run_timeout else {
        return handle(task).await;
    };
    if tokio::time::timeout(limit, handle(task)).await.is_err() {
        tracing::warn!(
            session_id = %bound.session_id,
            kind = bound.kind.label(),
            id = %bound.id,
            run_secs = limit.as_secs(),
            "cancelling a task that ran too long"
        );
        expire(bound, RUN).await;
    }
}

async fn expire(store: &dyn EventStore, bound: &TaskBound, message: &str) {
    let command = CommandPayload::settle(
        bound.kind,
        bound.id.clone(),
        bound.attempt,
        SettleError::new(ErrorInfo::new(ErrorCode::DeadlineExceeded, message), true),
    );
    let result = execute(
        store,
        ExecuteInput {
            session_id: bound.session_id.clone(),
            caller: Caller::System {
                tenant_id: bound.tenant_id.clone(),
            },
            command,
            span: bound.span.child("expire"),
        },
        &ConflictRetry::default(),
    )
    .await;
    if let Err(err) = result {
        tracing::error!(
            session_id = %bound.session_id,
            id = %bound.id,
            error = %err,
            "failed to settle a lapsed task"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use super::*;

    struct Task(Option<TaskBound>);

    impl BoundedTask for Task {
        fn bound(&self) -> Option<TaskBound> {
            self.0.clone()
        }
    }

    fn bound(queue: Option<u64>, run: Option<u64>, waited_ms: i64) -> Option<TaskBound> {
        Some(TaskBound {
            tenant_id: "t".into(),
            session_id: "s".into(),
            kind: EffectKind::ToolCall,
            id: "tc-1".into(),
            attempt: Some(0),
            enqueued_at: Utc::now() - chrono::Duration::milliseconds(waited_ms),
            queue_timeout: queue.map(Duration::from_millis),
            run_timeout: run.map(Duration::from_millis),
            span: SpanContext::root(),
        })
    }

    async fn drive(task: Task, work: Duration) -> (bool, Option<&'static str>) {
        let ran = Arc::new(AtomicBool::new(false));
        let expired: Arc<Mutex<Option<&'static str>>> = Arc::new(Mutex::new(None));
        let (r, e) = (ran.clone(), expired.clone());
        run_bounded(
            task,
            |_| {
                let r = r.clone();
                async move {
                    tokio::time::sleep(work).await;
                    r.store(true, Ordering::SeqCst);
                }
            },
            |_, message| {
                let e = e.clone();
                async move { *e.lock().unwrap() = Some(message) }
            },
        )
        .await;
        let out = *expired.lock().unwrap();
        (ran.load(Ordering::SeqCst), out)
    }

    #[tokio::test]
    async fn work_inside_both_bounds_runs_and_settles_itself() {
        let (ran, expired) = drive(Task(bound(Some(500), Some(500), 0)), Duration::ZERO).await;
        assert!(ran);
        assert_eq!(expired, None, "the work settles its own result");
    }

    #[tokio::test]
    async fn a_task_past_its_queue_bound_never_runs() {
        let (ran, expired) = drive(Task(bound(Some(10), Some(500), 200)), Duration::ZERO).await;
        assert!(!ran, "it was stale before an executor reached it");
        assert_eq!(expired, Some(QUEUED));
    }

    #[tokio::test]
    async fn work_past_its_run_bound_is_cancelled() {
        let (ran, expired) = drive(
            Task(bound(Some(500), Some(20), 0)),
            Duration::from_millis(5_000),
        )
        .await;
        assert!(!ran, "the future was dropped before it finished");
        assert_eq!(expired, Some(RUN));
    }

    #[tokio::test]
    async fn a_long_wait_does_not_shorten_the_run() {
        let (ran, expired) = drive(
            Task(bound(Some(500), Some(200), 400)),
            Duration::from_millis(100),
        )
        .await;
        assert!(ran, "400ms queued, and still a full run bound to work in");
        assert_eq!(expired, None);
    }

    #[tokio::test]
    async fn an_unbounded_run_is_never_cancelled() {
        let (ran, expired) =
            drive(Task(bound(Some(500), None, 0)), Duration::from_millis(50)).await;
        assert!(ran);
        assert_eq!(expired, None);
    }

    #[tokio::test]
    async fn plumbing_no_effect_waits_on_runs_unbounded() {
        let (ran, expired) = drive(Task(None), Duration::from_millis(50)).await;
        assert!(ran);
        assert_eq!(expired, None, "nothing is holding a result to settle");
    }

    #[tokio::test]
    async fn a_stamp_in_the_future_is_no_wait() {
        let (ran, _) = drive(Task(bound(Some(10), Some(500), -5_000)), Duration::ZERO).await;
        assert!(ran);
    }
}
