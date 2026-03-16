import { createFileRoute, Link } from '@tanstack/react-router'
import type { DerivedState } from '@substructure.ai/client/types'
import { EventList } from '#/components/EventList.tsx'
import { useSessionEvents } from '#/hooks/useSessionEvents.ts'
import { formatDuration } from '#/components/events/format.ts'

export const Route = createFileRoute('/sessions/$sessionId')({
  component: SessionDetailPage,
})

function formatStatus(status: DerivedState['status']): string {
  if (typeof status === 'string') return status.charAt(0).toUpperCase() + status.slice(1)
  if (typeof status === 'object' && 'interrupted' in status) return 'Interrupted'
  return String(status)
}

function formatTotalCost(derived: DerivedState): string {
  const own = parseFloat(derived.cost || '0')
  const sub = parseFloat(derived.sub_agent_cost || '0')
  return `$${own + sub}`
}

function SessionOverview({ sessionId, derived, firstEventAt, lastEventAt }: {
  sessionId: string
  derived?: DerivedState
  firstEventAt?: string
  lastEventAt?: string
}) {
  const durationMs = firstEventAt && lastEventAt
    ? new Date(lastEventAt).getTime() - new Date(firstEventAt).getTime()
    : null

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-4 py-3">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-[var(--color-text-secondary)]">Session ID</dt>
        <dd className="font-mono text-xs text-[var(--color-text)]">{sessionId}</dd>
        <dt className="text-[var(--color-text-secondary)]">Status</dt>
        <dd className="text-[var(--color-text)]">{derived ? formatStatus(derived.status) : '-'}</dd>
        <dt className="text-[var(--color-text-secondary)]">Agent</dt>
        <dd className="text-[var(--color-text)]">{derived?.agent_id ?? '-'}</dd>
        <dt className="text-[var(--color-text-secondary)]">Cost</dt>
        <dd className="text-[var(--color-text)]">{derived ? formatTotalCost(derived) : '-'}</dd>
        {firstEventAt && <>
          <dt className="text-[var(--color-text-secondary)]">First Event</dt>
          <dd className="font-mono text-xs text-[var(--color-text)]">{new Date(firstEventAt).toLocaleString()}</dd>
        </>}
        {lastEventAt && <>
          <dt className="text-[var(--color-text-secondary)]">Last Event</dt>
          <dd className="font-mono text-xs text-[var(--color-text)]">{new Date(lastEventAt).toLocaleString()}</dd>
        </>}
        {durationMs != null && <>
          <dt className="text-[var(--color-text-secondary)]">Duration</dt>
          <dd className="font-mono text-xs text-[var(--color-text)]">{formatDuration(0, durationMs)}</dd>
        </>}
      </dl>
    </div>
  )
}

function SessionDetailPage() {
  const { sessionId } = Route.useParams()
  const { items, isLoading, derived, firstEventAt, lastEventAt } = useSessionEvents(sessionId)

  return (
    <main className="mx-auto max-w-5xl px-4 pb-8 pt-8">
      <div className="mb-6">
        <Link to="/sessions" className="text-sm">
          &larr; All Sessions
        </Link>
      </div>

      <h2 className="mb-3 text-lg font-semibold">Overview</h2>
      <SessionOverview sessionId={sessionId} derived={derived} firstEventAt={firstEventAt} lastEventAt={lastEventAt} />

      <h2 className="mb-3 mt-6 text-lg font-semibold">Events</h2>
      {isLoading && <p className="text-[var(--color-text-secondary)]">Loading events...</p>}

      {!isLoading && items.length === 0 && (
        <p className="text-[var(--color-text-secondary)]">No events yet.</p>
      )}

      {items.length > 0 && <EventList items={items} />}
    </main>
  )
}
