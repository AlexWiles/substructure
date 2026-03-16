import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useInfiniteQuery } from '@tanstack/react-query'
import { adminClient } from '#/lib/api.ts'
import { Page, Breadcrumbs, Table, THead, TBody, Th, Td, Button } from '#/components/ui.tsx'

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
        limit: 20,
        cursor: pageParam,
      }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: 5000,
  })

  const navigate = useNavigate()
  const sessions = data?.pages.flatMap((p) => p.items) ?? []

  return (
    <Page>
      <Breadcrumbs crumbs={[
        { label: 'sessions', to: '/sessions' },
      ]} />

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
        <Table>
          <THead>
            <Th>Session ID</Th>
            <Th>Agent</Th>
            <Th>Tenant</Th>
            <Th align="right">Cost</Th>
            <Th>Last Activity</Th>
          </THead>
          <TBody>
            {sessions.map((item: SessionListItem) => (
              <tr
                key={item.summary.aggregate_id}
                className="cursor-pointer transition-colors hover:bg-[var(--color-hover)]"
                onClick={() => navigate({ to: '/sessions/$sessionId', params: { sessionId: item.summary.aggregate_id } })}
              >
                <Td>
                  <Link
                    to="/sessions/$sessionId"
                    params={{ sessionId: item.summary.aggregate_id }}
                  >
                    {item.summary.aggregate_id}
                  </Link>
                </Td>
                <Td>{item.state.agent_id ?? '-'}</Td>
                <Td secondary>{item.summary.tenant_id}</Td>
                <Td secondary align="right">
                  ${(parseFloat(item.state.cost || '0') + parseFloat(item.state.sub_agent_cost || '0')).toFixed(6)}
                </Td>
                <Td secondary>
                  {item.summary.last_event_at
                    ? new Date(item.summary.last_event_at).toLocaleString()
                    : '-'}
                </Td>
              </tr>
            ))}
          </TBody>
        </Table>
      )}

      {hasNextPage && (
        <div className="mt-4 text-center">
          <Button onClick={() => fetchNextPage()} disabled={isFetchingNextPage}>
            {isFetchingNextPage ? 'Loading...' : 'Load more'}
          </Button>
        </div>
      )}
    </Page>
  )
}
