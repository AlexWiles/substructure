import { memo } from 'react'
import { Loader2, CheckCircle2, XCircle, Circle, MessageSquare, Play, Square, Zap, Bot, Wrench, BrainCircuit, RotateCcw } from 'lucide-react'
import type { Event, EventPayload, DerivedState } from '@substructure.ai/client/types'
import { useSessionEvents } from '#/hooks/useSessionEvents.ts'
import { formatTime, formatDuration } from './format.ts'
import { isGroup } from './grouping.ts'
import type { EventGroup, GroupedItem } from './grouping.ts'
import { EventBody } from './EventDetail.tsx'
import { TreeRow } from './TimelineRow.tsx'

// ── Shared ──────────────────────────────────────────────────────────────────

const ICON_CLASS = 'text-[var(--color-text-secondary)]'

function eventIcon(type: EventPayload['type']): React.ReactNode {
  switch (type) {
    case 'message.new':               return <MessageSquare size={12} className={ICON_CLASS} />
    case 'session.created':           return <Play size={12} className={ICON_CLASS} />
    case 'session.done':              return <CheckCircle2 size={12} className="text-green-600 dark:text-green-400" />
    case 'session.cancelled':         return <Square size={12} className="text-red-500" />
    case 'session.interrupted':       return <Square size={12} className="text-amber-500" />
    case 'llm.call.requested':        return <BrainCircuit size={12} className={ICON_CLASS} />
    case 'llm.call.completed':        return <BrainCircuit size={12} className={ICON_CLASS} />
    case 'llm.call.errored':          return <XCircle size={12} className="text-red-500" />
    case 'tool.call.requested':       return <Wrench size={12} className={ICON_CLASS} />
    case 'tool.call.completed':       return <Wrench size={12} className={ICON_CLASS} />
    case 'tool.call.errored':         return <XCircle size={12} className="text-red-500" />
    case 'sub_agent.requested':       return <Bot size={12} className={ICON_CLASS} />
    case 'sub_agent.started':         return <Bot size={12} className={ICON_CLASS} />
    case 'sub_agent.turn_completed':  return <Bot size={12} className={ICON_CLASS} />
    case 'sub_agent.errored':         return <XCircle size={12} className="text-red-500" />
    case 'worker.decision.requested': return <Zap size={12} className={ICON_CLASS} />
    case 'worker.decision.completed': return <Zap size={12} className={ICON_CLASS} />
    case 'worker.decision.errored':   return <XCircle size={12} className="text-red-500" />
    case 'turn.started':              return <RotateCcw size={12} className={ICON_CLASS} />
    case 'turn.completed':            return <RotateCcw size={12} className={ICON_CLASS} />
    default:                          return <Circle size={4} className="text-[var(--color-text-secondary)] fill-current" />
  }
}

function Timestamp({ iso }: { iso: string }) {
  return (
    <span className="text-xs text-[var(--color-text-secondary)] whitespace-nowrap">
      {formatTime(iso)}
    </span>
  )
}

function eventSummary(payload: EventPayload): string {
  switch (payload.type) {
    case 'message.new':          return payload.message.role
    case 'session.created':      return `agent: ${payload.agent_id}`
    case 'session.done':         return 'completed'
    case 'session.cancelled':    return 'cancelled'
    case 'turn.started':
    case 'turn.completed':       return payload.turn_id
    case 'worker.state.updated': return ''
    default:                     return ''
  }
}

// ── ItemList ────────────────────────────────────────────────────────────────

export function ItemList({ items, depth = 0 }: { items: GroupedItem[]; depth?: number }) {
  return (
    <>
      {items.map((item) =>
        isGroup(item) ? (
          <GroupRow key={`group-${item.startEvent.aggregate_id}-${item.startEvent.sequence}`} group={item} depth={depth} />
        ) : (
          <EventRow key={`${item.aggregate_id}-${item.sequence}`} event={item} depth={depth} />
        ),
      )}
    </>
  )
}

// ── EventRow ────────────────────────────────────────────────────────────────

export const EventRow = memo(function EventRow({ event, depth = 0 }: { event: Event; depth?: number }) {
  return (
    <TreeRow
      depth={depth}
      icon={eventIcon(event.payload.type)}
      label={event.payload.type}
      summary={eventSummary(event.payload)}
      trailing={<Timestamp iso={event.occurred_at} />}
    >
      <div className="bg-[var(--color-bg)] px-4 py-2 mx-2 mb-1" style={{ marginLeft: `${32 + depth * 20}px` }}>
        <EventBody event={event} />
      </div>
    </TreeRow>
  )
})

// ── GroupRow ─────────────────────────────────────────────────────────────────

export const GroupRow = memo(function GroupRow({ group, depth = 0 }: { group: EventGroup; depth?: number }) {
  const errored = group.endEvent?.payload.type.endsWith('.errored') ?? false
  const inProgress = !group.endEvent
  const allEvents = [group.startEvent, ...group.innerEvents, ...(group.endEvent ? [group.endEvent] : [])]

  const subAgentSessionId =
    group.startEvent.payload.type === 'sub_agent.requested'
      ? group.startEvent.payload.session_id
      : null

  return (
    <TreeRow
      depth={depth}
      icon={<GroupStatusIcon inProgress={inProgress} errored={errored} />}
      label={group.rule.label}
      summary={group.rule.summarize(group.startEvent.payload, group.endEvent?.payload)}
      trailing={
        <>
          <Duration start={group.startEvent} end={group.endEvent} errored={errored} inProgress={inProgress} />
          <Timestamp iso={group.startEvent.occurred_at} />
        </>
      }
    >
      <ItemList items={allEvents.map(e => e as GroupedItem)} depth={depth + 1} />
      {subAgentSessionId && <SubAgentGroup sessionId={subAgentSessionId} depth={depth + 1} />}
    </TreeRow>
  )
})

// ── SubAgentGroup ───────────────────────────────────────────────────────────

function SubAgentGroup({ sessionId, depth = 0 }: { sessionId: string; depth?: number }) {
  const { items, isDone, isLoading, derived } = useSessionEvents(sessionId)

  return (
    <TreeRow
      depth={depth}
      icon={
        isDone
          ? <CheckCircle2 size={12} className="text-green-600 dark:text-green-400" />
          : <Loader2 size={12} className="animate-spin text-[var(--color-accent)]" />
      }
      label={derived?.agent_id ?? 'sub-agent'}
      summary={derived && formatDerivedSummary(derived)}
      trailing={
        <SubAgentTrailing items={items} isDone={isDone} />
      }
    >
      {isLoading ? (
        <div className="flex items-center gap-2 py-2 text-xs text-[var(--color-text-secondary)]" style={{ paddingLeft: `${32 + depth * 20}px` }}>
          <Loader2 size={12} className="animate-spin" /> Loading&hellip;
        </div>
      ) : (
        <ItemList items={items} depth={depth + 1} />
      )}
    </TreeRow>
  )
}

function formatTotalCost(derived: DerivedState): string {
  const own = parseFloat(derived.cost || '0')
  const sub = parseFloat(derived.sub_agent_cost || '0')
  return `$${own + sub}`
}

function formatDerivedSummary(derived: DerivedState): string {
  const status = formatStatus(derived.status)
  return [status, formatTotalCost(derived)].join(' · ')
}

function formatStatus(status: DerivedState['status']): string {
  if (typeof status === 'string') return status.charAt(0).toUpperCase() + status.slice(1)
  if (typeof status === 'object' && 'interrupted' in status) return 'Interrupted'
  return String(status)
}

function SubAgentTrailing({ items, isDone }: { items: GroupedItem[]; isDone: boolean }) {
  const firstItem = items[0] as GroupedItem | undefined
  const firstEvent = firstItem ? (isGroup(firstItem) ? firstItem.startEvent : firstItem) : undefined

  const lastItem = items[items.length - 1] as GroupedItem | undefined
  const lastEvent = lastItem ? (isGroup(lastItem) ? (lastItem.endEvent ?? lastItem.startEvent) : lastItem) : undefined

  const duration = firstEvent && lastEvent && isDone
    ? formatDuration(new Date(firstEvent.occurred_at).getTime(), new Date(lastEvent.occurred_at).getTime())
    : 'in progress\u2026'

  return (
    <>
      <span className={`text-xs font-mono whitespace-nowrap ${isDone ? 'text-[var(--color-text-secondary)]' : 'text-[var(--color-accent)]'}`}>
        {duration}
      </span>
      {firstEvent && <Timestamp iso={firstEvent.occurred_at} />}
    </>
  )
}

// ── Small shared pieces ─────────────────────────────────────────────────────

function GroupStatusIcon({ inProgress, errored }: { inProgress: boolean; errored: boolean }) {
  if (inProgress) return <Loader2 size={12} className="animate-spin text-[var(--color-accent)]" />
  if (errored)    return <XCircle size={12} className="text-red-500" />
  return <CheckCircle2 size={12} className="text-[var(--color-text-secondary)]" />
}

function Duration({ start, end, errored, inProgress }: { start: Event; end?: Event; errored: boolean; inProgress: boolean }) {
  const text = end
    ? formatDuration(new Date(start.occurred_at).getTime(), new Date(end.occurred_at).getTime())
    : 'in progress\u2026'
  const color = errored
    ? 'text-red-600 dark:text-red-400'
    : inProgress ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'
  return <span className={`text-xs font-mono whitespace-nowrap ${color}`}>{text}</span>
}
