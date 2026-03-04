//! OpenTelemetry exporter actor.
//!
//! Subscribes to the event store broadcast and exports persisted events as
//! OTel spans via OTLP. Events from a single `execute()` call share a span_id
//! and carry wall-clock `start_time`/`end_time` for accurate trace timing.

use std::borrow::Cow;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef};
use tokio::task::AbortHandle;

use opentelemetry::trace::{SpanKind, Status, TraceFlags, TraceState};
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_sdk::trace::{SpanData, SpanEvents, SpanExporter, SpanLinks};

use crate::runtime::event_store::{
    reconstruct_span_summaries, Event, EventBatch, EventStore, SpanSummary,
};

const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum OtelMsg {
    Events(EventBatch),
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct OtelExporterActor;

pub struct OtelExporterState {
    exporter: opentelemetry_otlp::SpanExporter,
    pending: Vec<SpanData>,
    scope: InstrumentationScope,
    timer_handle: Option<AbortHandle>,
    myself: ActorRef<OtelMsg>,
}

pub struct OtelExporterArgs {
    pub exporter: opentelemetry_otlp::SpanExporter,
    pub service_name: String,
}

impl OtelExporterState {
    fn schedule_tick(&mut self) {
        if let Some(handle) = self.timer_handle.take() {
            handle.abort();
        }
        let actor = self.myself.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(FLUSH_INTERVAL).await;
            let _ = actor.send_message(OtelMsg::Tick);
        });
        self.timer_handle = Some(handle.abort_handle());
    }

    async fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        let otel_spans: Vec<SpanData> = self.pending.drain(..).collect();
        tracing::debug!(count = otel_spans.len(), "exporting otel spans");
        if let Err(e) = self.exporter.export(otel_spans).await {
            tracing::warn!(error = ?e, "otel export failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Span building
// ---------------------------------------------------------------------------

fn datetime_to_system_time(dt: DateTime<Utc>) -> SystemTime {
    let duration = dt.signed_duration_since(DateTime::UNIX_EPOCH);
    if let Ok(std_duration) = duration.to_std() {
        SystemTime::UNIX_EPOCH + std_duration
    } else {
        SystemTime::UNIX_EPOCH
    }
}

/// Convert a `SpanSummary` into OTel `SpanData`.
fn span_summary_to_otel(summary: &SpanSummary, scope: &InstrumentationScope) -> SpanData {
    let trace_id = opentelemetry::trace::TraceId::from_bytes(summary.span.trace_id.as_bytes());
    let span_id = opentelemetry::trace::SpanId::from_bytes(summary.span.span_id.as_bytes());
    let parent_span_id = summary
        .span
        .parent_span_id
        .map(|id| opentelemetry::trace::SpanId::from_bytes(id.as_bytes()))
        .unwrap_or(opentelemetry::trace::SpanId::INVALID);

    let span_context = opentelemetry::trace::SpanContext::new(
        trace_id,
        span_id,
        TraceFlags::SAMPLED,
        false,
        TraceState::NONE,
    );

    let mut attributes: Vec<KeyValue> = summary
        .attributes
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();

    let status = match &summary.error {
        Some(msg) => {
            attributes.push(KeyValue::new("error.message", msg.clone()));
            Status::error(msg.clone())
        }
        None => Status::Ok,
    };

    SpanData {
        span_context,
        parent_span_id,
        parent_span_is_remote: false,
        span_kind: SpanKind::Internal,
        name: Cow::Owned(summary.name.clone()),
        start_time: datetime_to_system_time(summary.start_time),
        end_time: datetime_to_system_time(summary.end_time),
        attributes,
        dropped_attributes_count: 0,
        events: SpanEvents::default(),
        links: SpanLinks::default(),
        status,
        instrumentation_scope: scope.clone(),
    }
}

// ---------------------------------------------------------------------------
// Actor impl
// ---------------------------------------------------------------------------

impl Actor for OtelExporterActor {
    type Msg = OtelMsg;
    type State = OtelExporterState;
    type Arguments = OtelExporterArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut state = OtelExporterState {
            exporter: args.exporter,
            pending: Vec::new(),
            scope: InstrumentationScope::builder(args.service_name).build(),
            timer_handle: None,
            myself,
        };
        state.schedule_tick();
        Ok(state)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            OtelMsg::Events(batch) => {
                let event_refs: Vec<&Event> = batch.iter().map(|e| e.as_ref()).collect();
                let summaries = reconstruct_span_summaries(&event_refs);
                let spans = summaries
                    .iter()
                    .map(|s| span_summary_to_otel(s, &state.scope));
                state.pending.extend(spans);
            }
            OtelMsg::Tick => {
                state.flush().await;
                state.schedule_tick();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Spawn helper
// ---------------------------------------------------------------------------

pub async fn spawn_otel_exporter(
    endpoint: &str,
    service_name: String,
    supervisor: ActorCell,
    store: &dyn EventStore,
) -> Result<ActorRef<OtelMsg>, Box<dyn std::error::Error>> {
    use opentelemetry_otlp::WithExportConfig;

    let resource = opentelemetry_sdk::Resource::builder()
        .with_attribute(KeyValue::new("service.name", service_name.clone()))
        .build();

    let mut exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    exporter.set_resource(&resource);

    let (actor_ref, _) = Actor::spawn_linked(
        Some("otel-exporter".to_string()),
        OtelExporterActor,
        OtelExporterArgs {
            exporter,
            service_name,
        },
        supervisor,
    )
    .await?;

    store
        .events()
        .subscribe(actor_ref.clone(), |batch| Some(OtelMsg::Events(batch)));

    Ok(actor_ref)
}
