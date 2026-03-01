//! OpenTelemetry exporter actor.
//!
//! Receives `ExecutionRecord`s from aggregate actors (inline instrumentation)
//! and exports them as OTel spans via OTLP. Each record maps 1:1 to an OTel
//! span with real wall-clock start/end times.

use std::borrow::Cow;
use std::time::Duration;

use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef};
use tokio::task::AbortHandle;

use opentelemetry::trace::{SpanKind, Status, TraceFlags, TraceState};
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_sdk::trace::{SpanData, SpanEvents, SpanExporter, SpanLinks};

use crate::runtime::aggregate_actor::ExecutionRecord;

const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub enum OtelMsg {
    Record(ExecutionRecord),
    Tick,
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

pub struct OtelExporterActor;

pub struct OtelExporterState {
    exporter: opentelemetry_otlp::SpanExporter,
    pending: Vec<ExecutionRecord>,
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

        let records: Vec<ExecutionRecord> = self.pending.drain(..).collect();
        let otel_spans: Vec<SpanData> = records
            .into_iter()
            .map(|r| to_span_data(r, &self.scope))
            .collect();

        tracing::debug!(count = otel_spans.len(), "exporting otel spans");
        if let Err(e) = self.exporter.export(otel_spans).await {
            tracing::warn!(error = ?e, "otel export failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

fn to_span_data(record: ExecutionRecord, scope: &InstrumentationScope) -> SpanData {
    let ctx = &record.span_context;
    let trace_id = opentelemetry::trace::TraceId::from_bytes(ctx.trace_id.as_bytes());
    let span_id = opentelemetry::trace::SpanId::from_bytes(ctx.span_id.as_bytes());
    let parent_span_id = ctx
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

    // Name: span context name (set to triggering event type), or first produced event type
    let name = ctx
        .name
        .clone()
        .or_else(|| {
            record
                .produced_events
                .first()
                .map(|e| e.event_type.clone())
        })
        .unwrap_or_else(|| "execute".to_string());

    let mut attributes = vec![
        KeyValue::new("aggregate.type", record.aggregate_type.to_string()),
        KeyValue::new("aggregate.id", record.aggregate_id.to_string()),
    ];
    for (i, event) in record.produced_events.iter().enumerate() {
        attributes.push(KeyValue::new(
            format!("event.{i}.type"),
            event.event_type.clone(),
        ));
    }

    SpanData {
        span_context,
        parent_span_id,
        parent_span_is_remote: false,
        span_kind: SpanKind::Internal,
        name: Cow::Owned(name),
        start_time: record.start_time,
        end_time: record.end_time,
        attributes,
        dropped_attributes_count: 0,
        events: SpanEvents::default(),
        links: SpanLinks::default(),
        status: Status::Ok,
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
            OtelMsg::Record(record) => {
                state.pending.push(record);
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

    Ok(actor_ref)
}
