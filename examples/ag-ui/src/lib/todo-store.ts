// A tiny framework-agnostic store for the demo's to-do list: a module-global
// singleton the frontend tools write to and TodoPanel reads via
// useSyncExternalStore. The same shape as the old color store, over a
// collection — every mutation swaps in a NEW array so useSyncExternalStore sees
// the change.

export type Todo = { id: string; title: string; done: boolean };

let seq = 0;
const nextId = (): string => `todo-${++seq}`;

let state: Todo[] = [
    { id: nextId(), title: "Check me off — or ask the agent to", done: false },
    { id: nextId(), title: "Try: “add milk, eggs, and bread”", done: false },
];
const listeners = new Set<() => void>();

export function getTodos(): Todo[] {
    return state;
}

/** Append a task and notify subscribers; returns the created item. */
export function addTodo(title: string): Todo {
    const todo: Todo = { id: nextId(), title: title.trim(), done: false };
    state = [...state, todo];
    emit();
    return todo;
}

/** Set (or flip) a task's done flag. Returns the updated item, or null if the
 *  id isn't found. */
export function toggleTodo(id: string, done?: boolean): Todo | null {
    const cur = state.find((t) => t.id === id);
    if (!cur) return null;
    const updated: Todo = { ...cur, done: done ?? !cur.done };
    state = state.map((t) => (t.id === id ? updated : t));
    emit();
    return updated;
}

/** Remove a task. Returns true if one was removed. */
export function removeTodo(id: string): boolean {
    const next = state.filter((t) => t.id !== id);
    if (next.length === state.length) return false;
    state = next;
    emit();
    return true;
}

/** Remove every completed task. Returns how many were removed. */
export function clearCompleted(): number {
    const next = state.filter((t) => !t.done);
    const removed = state.length - next.length;
    if (removed) {
        state = next;
        emit();
    }
    return removed;
}

function emit(): void {
    for (const notify of listeners) notify();
}

export function subscribeTodos(onChange: () => void): () => void {
    listeners.add(onChange);
    return () => {
        listeners.delete(onChange);
    };
}
