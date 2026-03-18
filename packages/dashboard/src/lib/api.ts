import { AdminClient } from '@substructure.ai/sdk/admin'

export const adminClient = new AdminClient({
  baseUrl: 'http://localhost:8080',
})

export { AdminClient }
export type {
  SessionListItem,
  SessionListResponse,
  SessionDetail,
  AggregateSummary,
  ListSessionsParams,
  SessionEventsParams,
  AggregateSort,
} from '@substructure.ai/sdk/admin'
