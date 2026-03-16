import { createFileRoute, Link } from '@tanstack/react-router'
import { useInfiniteQuery } from '@tanstack/react-query'
import { adminClient } from '#/lib/api.ts'
import { StatusBadge } from '#/components/StatusBadge.tsx'

import type { SessionListItem } from '#/lib/api.ts'

export const Route = createFileRoute('/sessions/')({ component: SessionsPage })

function SessionsPage() {
  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ['sessions'],
    queryFn: ({ pageParam }) =>
      adminClient.listSessions({
        top_level: true,
        sort: 'last_event_desc',
        limit: 50,
        cursor: pageParam,
      }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: 5000,
  })

  const sessions = data?.pages.flatMap((p) => p.items) ?? []

  return (
    <main className="mx-auto max-w-5xl px-4 pb-8 pt-8">
      <h1 className="mb-6 text-2xl font-bold">Sessions</h1>

      {isLoading && <p className="text-[var(--color-text-secondary)]">Loading sessions...</p>}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          Failed to load sessions: {(error as Error).message}
        </div>
      )}

      {!isLoading && sessions.length === 0 && (
        <p className="text-[var(--color-text-secondary)]">No sessions yet.</p>
      )}

      {sessions.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-bg)]">
                <th className="px-4 py-3 text-left font-semibold">Session ID</th>
                <th className="px-4 py-3 text-left font-semibold">Agent</th>
                <th className="px-4 py-3 text-left font-semibold">Status</th>
                <th className="px-4 py-3 text-left font-semibold">Tenant</th>
                <th className="px-4 py-3 text-left font-semibold">Last Activity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-bg)]">
              {sessions.map((item: SessionListItem) => (
                <tr
                  key={item.summary.aggregate_id}
                  className="transition-colors hover:bg-[var(--color-hover)]"
                >
                  <td className="px-4 py-3">
                    <Link
                      to="/sessions/$sessionId"
                      params={{ sessionId: item.summary.aggregate_id }}
                      className="font-mono text-xs"
                    >
                      {item.summary.aggregate_id}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    {item.state.agent_id ?? '-'}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={item.state.status} />
                  </td>
                  <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                    {item.summary.tenant_id}
                  </td>
                  <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                    {item.summary.last_event_at
                      ? new Date(item.summary.last_event_at).toLocaleString()
                      : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {hasNextPage && (
        <div className="mt-4 text-center">
          <button
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-sm text-[var(--color-text)] hover:bg-[var(--color-border)] disabled:opacity-50"
          >
            {isFetchingNextPage ? 'Loading...' : 'Load more'}
          </button>
        </div>
      )}
    </main>
  )
}
