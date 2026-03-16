import { useState } from 'react'
import Markdown from 'react-markdown'
import { Braces, Copy } from 'lucide-react'
import type { Event, EventPayload } from '@substructure.ai/client/types'

// ── Reusable detail layout ──────────────────────────────────────────────────

export function DetailGrid({ children }: { children: React.ReactNode }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {children}
    </dl>
  )
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-[var(--color-text-secondary)]">{label}</dt>
      <dd className="text-[var(--color-text)]">{children}</dd>
    </>
  )
}

function ErrorField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-[var(--color-text-secondary)]">{label}</dt>
      <dd className="text-red-600 dark:text-red-400">{children}</dd>
    </>
  )
}

function RetryableField({ retryable }: { retryable: boolean }) {
  return <Field label="Retryable">{retryable ? 'yes' : 'no'}</Field>
}

// ── Per-type formatted views ────────────────────────────────────────────────

function renderDetail(payload: EventPayload): React.ReactNode {
  switch (payload.type) {
    case 'message.new': {
      const { message: msg } = payload
      const content = msg.content ?? ''
      const hasToolCalls = msg.tool_calls && msg.tool_calls.length > 0
      return (
        <div className="space-y-2 text-sm">
          {content && (
            msg.role === 'tool' ? (
              <pre className="max-h-64 overflow-auto text-xs text-[var(--color-text-secondary)] whitespace-pre-wrap break-all">
                {tryFormatJson(content)}
              </pre>
            ) : (
              <div className="prose prose-sm dark:prose-invert max-w-none text-[var(--color-text)]">
                <Markdown>{content}</Markdown>
              </div>
            )
          )}
          {hasToolCalls && (
            <div className="space-y-1">
              {msg.tool_calls!.map((tc, i) => (
                <div key={i} className="flex items-center gap-2 text-xs font-mono text-[var(--color-text-secondary)]">
                  <span className="text-[var(--color-text)]">{tc.function.name}</span>
                  <span className="truncate">{tc.function.arguments}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )
    }

    case 'session.created':
      return (
        <DetailGrid>
          <Field label="Agent">{payload.agent_id}</Field>
          <Field label="Tenant">{payload.auth.tenant_id}</Field>
          {payload.ancestry && payload.ancestry.length > 0 && (
            <Field label="Ancestry">
              <span className="font-mono text-xs">{payload.ancestry.join(' \u2192 ')}</span>
            </Field>
          )}
        </DetailGrid>
      )

    case 'session.done':
      return (
        <div className="text-sm text-green-600 dark:text-green-400 font-medium">Session completed</div>
      )

    case 'session.cancelled':
      return (
        <DetailGrid>
          <ErrorField label="Status">Cancelled</ErrorField>
        </DetailGrid>
      )

    case 'llm.call.requested':
      return (
        <DetailGrid>
          <Field label="Model">{payload.request.model}</Field>
          <Field label="Client">{payload.llm_client}</Field>
          <Field label="Stream">{payload.stream ? 'yes' : 'no'}</Field>
          <Field label="Messages">{payload.request.messages.length}</Field>
          {payload.request.tools && payload.request.tools.length > 0 && (
            <Field label="Tools">{payload.request.tools.length}</Field>
          )}
        </DetailGrid>
      )

    case 'llm.call.completed':
      return (
        <DetailGrid>
          <Field label="Model">{payload.response.model}</Field>
          <Field label="Finish">{payload.response.finish_reason}</Field>
          {payload.response.cost && <Field label="Cost">${payload.response.cost}</Field>}
          {payload.response.content && (
            <Field label="Content">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <Markdown>{payload.response.content}</Markdown>
              </div>
            </Field>
          )}
          {payload.response.tool_calls.length > 0 && (
            <Field label="Tool calls">
              {payload.response.tool_calls.map(tc => tc.function.name).join(', ')}
            </Field>
          )}
        </DetailGrid>
      )

    case 'llm.call.errored':
      return (
        <DetailGrid>
          <ErrorField label="Error">{payload.error}</ErrorField>
          <RetryableField retryable={payload.retryable} />
        </DetailGrid>
      )

    case 'tool.call.requested':
      return (
        <DetailGrid>
          <Field label="Tool"><span className="font-medium">{payload.name}</span></Field>
          <Field label="Handler">{payload.handler}</Field>
          <Field label="Arguments">
            <span className="font-mono text-xs whitespace-pre-wrap break-all">{payload.arguments}</span>
          </Field>
        </DetailGrid>
      )

    case 'tool.call.completed':
      return (
        <DetailGrid>
          <Field label="Tool"><span className="font-medium">{payload.name}</span></Field>
          <Field label="Result">
            <span className="whitespace-pre-wrap break-all">
              {payload.result.length > 500 ? payload.result.slice(0, 500) + '\u2026' : payload.result}
            </span>
          </Field>
        </DetailGrid>
      )

    case 'tool.call.errored':
      return (
        <DetailGrid>
          <Field label="Tool"><span className="font-medium">{payload.name}</span></Field>
          <ErrorField label="Error">{payload.error}</ErrorField>
          <RetryableField retryable={payload.retryable} />
        </DetailGrid>
      )

    case 'worker.decision.requested':
      return (
        <DetailGrid>
          <Field label="Trigger">{payload.trigger.type}</Field>
        </DetailGrid>
      )

    case 'worker.decision.errored':
      return (
        <DetailGrid>
          <ErrorField label="Error">{payload.error}</ErrorField>
          <RetryableField retryable={payload.retryable} />
        </DetailGrid>
      )

    case 'sub_agent.requested':
      return (
        <DetailGrid>
          <Field label="Agent">{payload.agent_id}</Field>
          <Field label="Session"><span className="font-mono text-xs">{payload.session_id}</span></Field>
        </DetailGrid>
      )

    case 'sub_agent.turn_completed':
      return (
        <DetailGrid>
          <Field label="Session"><span className="font-mono text-xs">{payload.session_id}</span></Field>
          <Field label="Cost">${payload.cost}</Field>
          {payload.token_usage && Object.keys(payload.token_usage).length > 0 && (
            <Field label="Tokens">
              {Object.entries(payload.token_usage).map(([k, v]) => `${k}: ${v}`).join(', ')}
            </Field>
          )}
        </DetailGrid>
      )

    case 'sub_agent.errored':
      return (
        <DetailGrid>
          <ErrorField label="Error">{payload.error}</ErrorField>
          <RetryableField retryable={payload.retryable} />
        </DetailGrid>
      )

    case 'session.interrupted':
      return (
        <DetailGrid>
          <Field label="Reason">{payload.reason}</Field>
          <Field label="Interrupt ID"><span className="font-mono text-xs">{payload.interrupt_id}</span></Field>
        </DetailGrid>
      )

    case 'turn.started':
      return (
        <DetailGrid>
          <Field label="Turn ID"><span className="font-mono text-xs">{payload.turn_id}</span></Field>
        </DetailGrid>
      )

    case 'turn.completed': {
      const { artifacts, turn_cost, turn_token_usage } = payload
      return (
        <div className="space-y-3 text-sm">
          <DetailGrid>
            <Field label="Turn ID"><span className="font-mono text-xs">{payload.turn_id}</span></Field>
            {turn_cost && <Field label="Turn Cost">${turn_cost}</Field>}
            {turn_token_usage && Object.keys(turn_token_usage).length > 0 && (
              <Field label="Token Usage">{Object.entries(turn_token_usage).map(([k, v]) => `${k}: ${v}`).join(', ')}</Field>
            )}
          </DetailGrid>
          {artifacts && artifacts.length > 0 && (
            <div className="space-y-3">
              <div className="text-xs font-bold text-[var(--color-text-secondary)] uppercase tracking-wide">Artifacts</div>
              {artifacts.map((artifact, i) => (
                <div key={i} className="rounded border border-[var(--color-border)] bg-[var(--color-bg)]">
                  {(artifact.name || artifact.description) && (
                    <div className="px-3 py-2 border-b border-[var(--color-border)]">
                      {artifact.name && <div className="font-semibold text-[var(--color-text)]">{artifact.name}</div>}
                      {artifact.description && <div className="text-xs text-[var(--color-text-secondary)]">{artifact.description}</div>}
                    </div>
                  )}
                  <div className="px-3 py-2 space-y-2">
                    {artifact.parts.map((part, j) => (
                      part.kind === 'text' ? (
                        <div key={j} className="prose prose-sm dark:prose-invert max-w-none text-[var(--color-text)]">
                          <Markdown>{part.text}</Markdown>
                        </div>
                      ) : (
                        <pre key={j} className="max-h-48 overflow-auto rounded bg-[var(--color-surface)] p-2 text-xs text-[var(--color-text-secondary)]">
                          {JSON.stringify(part.data, null, 2)}
                        </pre>
                      )
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )
    }

    default:
      return null
  }
}

// ── Event body with raw toggle ──────────────────────────────────────────────

function ActionButton({ icon, active, title, onClick }: { icon: React.ReactNode; active?: boolean; title: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`p-1 cursor-pointer transition-colors rounded ${
        active
          ? 'bg-[var(--color-surface)] text-[var(--color-text)]'
          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)]'
      }`}
    >
      {icon}
    </button>
  )
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function detailHeader(payload: EventPayload): string | null {
  switch (payload.type) {
    case 'message.new':              return `${capitalize(payload.message.role)}:`
    case 'session.created':          return `Session Created`
    case 'session.done':             return `Session Done`
    case 'session.cancelled':        return `Session Cancelled`
    case 'session.interrupted':      return `Interrupted: ${payload.reason}`
    case 'llm.call.requested':       return payload.request.model
    case 'llm.call.completed':       return `${payload.response.model} · ${payload.response.finish_reason ?? 'done'}`
    case 'llm.call.errored':         return `LLM Error`
    case 'tool.call.requested':      return payload.name
    case 'tool.call.completed':      return payload.name
    case 'tool.call.errored':        return `${payload.name} · Error`
    case 'sub_agent.requested':      return payload.agent_id
    case 'sub_agent.turn_completed':  return `Sub-Agent Turn · $${payload.cost}`
    case 'sub_agent.errored':        return `Sub-Agent Error`
    case 'worker.decision.requested': return `Decision: ${payload.trigger.type}`
    case 'worker.decision.errored':  return `Decision Error`
    case 'turn.started':             return `Turn Started`
    case 'turn.completed':           return payload.turn_cost ? `Turn Completed · $${payload.turn_cost}` : `Turn Completed`
    default:                         return null
  }
}

export function EventBody({ event }: { event: Event }) {
  const formatted = renderDetail(event.payload)
  const hasFormatted = formatted !== null
  const [raw, setRaw] = useState(!hasFormatted)
  const header = detailHeader(event.payload)

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        {header && <span className="text-xs font-semibold text-[var(--color-text)]">{header}</span>}
        <span className="flex-1" />
        <div className="flex items-center gap-0.5">
        <ActionButton
          icon={<Copy size={13} />}
          title="Copy JSON"
          onClick={() => navigator.clipboard.writeText(JSON.stringify(event, null, 2))}
        />
        {hasFormatted && (
          <ActionButton
            icon={<Braces size={13} />}
            active={raw}
            title={raw ? 'Show formatted' : 'Show raw JSON'}
            onClick={() => setRaw(!raw)}
          />
        )}
        </div>
      </div>
      {raw ? (
        <pre className="max-h-96 overflow-auto bg-[var(--color-bg)] p-3 text-xs text-[var(--color-text-secondary)]">
          {JSON.stringify(event, null, 2)}
        </pre>
      ) : (
        formatted
      )}
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function tryFormatJson(s: string): string {
  try { return JSON.stringify(JSON.parse(s), null, 2) } catch { return s }
}
