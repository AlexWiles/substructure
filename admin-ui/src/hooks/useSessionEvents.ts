import { useEffect, useReducer } from 'react'
import type { Event } from '@substructure.ai/client/types'
import { adminClient } from '#/lib/api.ts'
import { applyEvent, applyEvents, initialState as initialEventState } from '#/components/events/reducer.ts'
import type { SessionEventState } from '#/components/events/reducer.ts'

// ── State machine ───────────────────────────────────────────────────────────

type Status = 'loading' | 'streaming' | 'done' | 'error'

interface State {
  events: SessionEventState
  status: Status
  error: Error | null
}

type Action =
  | { type: 'event'; event: Event }
  | { type: 'batch'; events: Event[] }
  | { type: 'stream_ended' }
  | { type: 'error'; error: Error }
  | { type: 'reset' }

function initialState(): State {
  return { events: initialEventState(), status: 'loading', error: null }
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'event':
      return {
        ...state,
        events: applyEvent(state.events, action.event),
        status: 'streaming',
      }
    case 'batch':
      return {
        ...state,
        events: applyEvents(initialEventState(), action.events),
        status: 'done',
      }
    case 'stream_ended':
      return { ...state, status: 'done' }
    case 'error':
      return { ...state, status: 'error', error: action.error }
    case 'reset':
      return initialState()
  }
}

// ── Hook ────────────────────────────────────────────────────────────────────

export function useSessionEvents(sessionId: string) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)

  useEffect(() => {
    const abort = new AbortController()
    dispatch({ type: 'reset' })

    streamEvents(sessionId, abort.signal, dispatch)

    return () => { abort.abort() }
  }, [sessionId])

  return {
    items: state.events.items,
    isDone: state.events.isDone,
    isLoading: state.status === 'loading',
    isStreaming: state.status === 'streaming',
    error: state.error,
    derived: state.events.derived,
    firstEventAt: state.events.firstEventAt,
    lastEventAt: state.events.lastEventAt,
  }
}

// ── Async logic (outside React) ─────────────────────────────────────────────

async function streamEvents(
  sessionId: string,
  signal: AbortSignal,
  dispatch: (action: Action) => void,
) {
  try {
    const stream = adminClient.streamSessionEvents(sessionId, undefined, { signal })
    for await (const event of stream) {
      if (signal.aborted) return
      dispatch({ type: 'event', event })
    }
    dispatch({ type: 'stream_ended' })
  } catch (err) {
    if (signal.aborted) return
    dispatch({ type: 'error', error: err as Error })
    await fetchFallback(sessionId, dispatch)
  }
}

async function fetchFallback(
  sessionId: string,
  dispatch: (action: Action) => void,
) {
  try {
    const events = await adminClient.getSessionEvents(sessionId)
    dispatch({ type: 'batch', events })
  } catch {
    // error state already set from the stream failure
  }
}
