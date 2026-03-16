import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useRef, useCallback } from 'react'
import type { DerivedState } from '@substructure.ai/client/types'
import { EventList } from '#/components/EventList.tsx'
import { KeyValue, Panel } from '#/components/ui.tsx'
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
    <Panel>
      <KeyValue label="Session" value={sessionId} />
      <KeyValue label="Status" value={derived ? formatStatus(derived.status) : '-'} />
      <KeyValue label="Agent" value={derived?.agent_id ?? '-'} />
      <KeyValue label="Cost" value={derived ? formatTotalCost(derived) : '-'} />
      {firstEventAt && <KeyValue label="Started" value={new Date(firstEventAt).toLocaleString()} />}
      {durationMs != null && <KeyValue label="Duration" value={formatDuration(0, durationMs)} />}
    </Panel>
  )
}

function useAutoScroll(itemCount: number) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const stickRef = useRef(true)
  const readyRef = useRef(false)
  const prevCountRef = useRef(itemCount)

  // Wait 2s after mount before enabling, so the historical replay settles
  useEffect(() => {
    const id = setTimeout(() => { readyRef.current = true }, 2000)
    return () => { clearTimeout(id); readyRef.current = false }
  }, [])

  const onScroll = useCallback(() => {
    const el = document.documentElement
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 64
    stickRef.current = atBottom
  }, [])

  useEffect(() => {
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [onScroll])

  useEffect(() => {
    const prev = prevCountRef.current
    prevCountRef.current = itemCount

    if (readyRef.current && itemCount > prev && stickRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [itemCount])

  return bottomRef
}

function SessionDetailPage() {
  const { sessionId } = Route.useParams()
  const { items, isLoading, isStreaming, derived, firstEventAt, lastEventAt } = useSessionEvents(sessionId)
  const bottomRef = useAutoScroll(items.length)

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
      <div ref={bottomRef} />
    </main>
  )
}
