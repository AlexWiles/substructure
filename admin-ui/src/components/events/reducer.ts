import type { Event, DerivedState } from '@substructure.ai/client/types'
import { GROUPING_RULES } from './grouping.ts'
import type { GroupingRule, EventGroup, GroupedItem } from './grouping.ts'

// ── Rule indexes (built once) ───────────────────────────────────────────────

const startTypeToRule = new Map<string, GroupingRule>()
const endTypeToRule = new Map<string, GroupingRule>()
for (const rule of GROUPING_RULES) {
  startTypeToRule.set(rule.startType, rule)
  for (const et of rule.endTypes) {
    endTypeToRule.set(et, rule)
  }
}

// ── State ───────────────────────────────────────────────────────────────────

export interface SessionEventState {
  /** Ordered list of top-level items for rendering. */
  items: GroupedItem[]
  /** Correlation key → index in `items` for open (in-progress) groups. */
  openGroups: Map<string, number>
  /** Correlation keys in insertion order, for orphan assignment. */
  openOrder: string[]
  /** Dedup set: "aggregateId:sequence" */
  seen: Set<string>
  isDone: boolean
  /** Latest derived state from the most recent event. */
  derived?: DerivedState
  tenantId?: string
  firstEventAt?: string
  lastEventAt?: string
  /** The root session's aggregate_id (set from the first event). */
  rootSessionId?: string
}

export function initialState(): SessionEventState {
  return {
    items: [],
    openGroups: new Map(),
    openOrder: [],
    seen: new Set(),
    isDone: false,
  }
}

// ── Reducer ─────────────────────────────────────────────────────────────────

function replaceItem(items: GroupedItem[], index: number, item: GroupedItem): GroupedItem[] {
  const next = items.slice()
  next[index] = item
  return next
}

function appendToGroup(items: GroupedItem[], index: number, event: Event): GroupedItem[] {
  const group = items[index] as EventGroup
  return replaceItem(items, index, { ...group, innerEvents: [...group.innerEvents, event] })
}

function findOwningGroup(state: SessionEventState, event: Event): number | undefined {
  for (const rule of GROUPING_RULES) {
    const corrKey = rule.correlationId(event.payload)
    if (corrKey != null) {
      const index = state.openGroups.get(corrKey)
      if (index != null) return index
    }
  }
  return undefined
}

export function applyEvent(state: SessionEventState, event: Event): SessionEventState {
  const dedupKey = `${event.aggregate_id}:${event.sequence}`
  if (state.seen.has(dedupKey)) return state

  const seen = new Set(state.seen)
  seen.add(dedupKey)

  const { type } = event.payload
  const isDone = state.isDone || type === 'session.done'
  const derived = event.derived ?? state.derived
  const tenantId = state.tenantId ?? event.tenant_id
  const firstEventAt = state.firstEventAt ?? event.occurred_at
  const lastEventAt = event.occurred_at
  const rootSessionId = state.rootSessionId ?? event.aggregate_id

  // ── Descendant event → skip (each SubAgentGroup has its own stream) ──
  if (rootSessionId !== event.aggregate_id) {
    return { ...state, seen, rootSessionId }
  }

  // ── Start event → open a new group ────────────────────────────────────
  const startRule = startTypeToRule.get(type)
  if (startRule) {
    const corrKey = startRule.correlationId(event.payload)
    if (corrKey != null) {
      const group: EventGroup = {
        kind: 'group',
        rule: startRule,
        startEvent: event,
        endEvent: undefined,
        innerEvents: [],
      }
      const openGroups = new Map(state.openGroups)
      openGroups.set(corrKey, state.items.length)
      return {
        items: [...state.items, group],
        openGroups,
        openOrder: [...state.openOrder, corrKey],
        seen,
        isDone,
        derived,
        tenantId,
        firstEventAt,
        lastEventAt,
        rootSessionId,
      }
    }
  }

  // ── End event → close an open group ───────────────────────────────────
  const endRule = endTypeToRule.get(type)
  if (endRule) {
    const corrKey = endRule.correlationId(event.payload)
    if (corrKey != null) {
      const index = state.openGroups.get(corrKey)
      if (index != null) {
        const group = state.items[index] as EventGroup
        const updated: EventGroup = { ...group, endEvent: event }
        const openGroups = new Map(state.openGroups)
        openGroups.delete(corrKey)
        return {
          items: replaceItem(state.items, index, updated),
          openGroups,
          openOrder: state.openOrder.filter(k => k !== corrKey),
          seen,
          isDone,
          derived,
          tenantId,
        firstEventAt,
          lastEventAt,
          rootSessionId,
        }
      }
    }
  }

  // ── Inner event → attach to an open group ─────────────────────────────
  if (state.openOrder.length > 0) {
    const groupIndex = findOwningGroup(state, event) ?? state.openGroups.get(state.openOrder.at(-1)!)!
    return {
      ...state,
      items: appendToGroup(state.items, groupIndex, event),
      seen,
      isDone,
      derived,
      firstEventAt,
      lastEventAt,
      rootSessionId,
    }
  }

  // ── Standalone event ──────────────────────────────────────────────────
  return {
    ...state,
    items: [...state.items, event],
    seen,
    isDone,
    derived,
    firstEventAt,
    lastEventAt,
    rootSessionId,
  }
}

/** Batch-apply events (for fallback fetch or initial load). */
export function applyEvents(state: SessionEventState, events: Event[]): SessionEventState {
  return events.reduce(applyEvent, state)
}
