import { useState, useSyncExternalStore } from "react";

import { addTodo, clearCompleted, getTodos, removeTodo, subscribeTodos, toggleTodo } from "../lib/todo-store";

// A checklist bound to the shared todo store: the agent's add_todo / toggle_todo
// / remove_todo / clear_completed tools mutate it, and you can edit it yourself.
// Both paths flow through the same store, so the UI stays in sync either way.
export function TodoPanel() {
    const todos = useSyncExternalStore(subscribeTodos, getTodos, getTodos);
    const [draft, setDraft] = useState("");

    const remaining = todos.filter((t) => !t.done).length;
    const hasDone = todos.some((t) => t.done);

    return (
        <aside className="todopanel">
            <div className="todopanel-title">To-do</div>

            <form
                className="todoadd"
                onSubmit={(e) => {
                    e.preventDefault();
                    const title = draft.trim();
                    if (!title) return;
                    addTodo(title);
                    setDraft("");
                }}
            >
                <input
                    className="todoinput"
                    placeholder="Add a task…"
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                />
                <button type="submit" className="todoaddbtn">
                    Add
                </button>
            </form>

            <ul className="todolist">
                {todos.length === 0 ? (
                    <li className="todoempty">Nothing here yet.</li>
                ) : (
                    todos.map((t) => (
                        <li key={t.id} className="todoitem" data-done={t.done}>
                            <label className="todocheck">
                                <input type="checkbox" checked={t.done} onChange={() => toggleTodo(t.id)} />
                                <span className="todotitle">{t.title}</span>
                            </label>
                            <button
                                type="button"
                                className="tododel"
                                aria-label="Delete task"
                                onClick={() => removeTodo(t.id)}
                            >
                                ×
                            </button>
                        </li>
                    ))
                )}
            </ul>

            <div className="todofoot">
                <span>
                    {remaining} {remaining === 1 ? "task" : "tasks"} left
                </span>
                {hasDone ? (
                    <button type="button" className="todoclear" onClick={() => clearCompleted()}>
                        Clear completed
                    </button>
                ) : null}
            </div>

            <p className="panelhint">
                The agent's <code>add_todo</code>, <code>toggle_todo</code>, and <code>clear_completed</code> drive
                this list (“add milk and eggs”, “check off the first one”); edit it yourself and ask “what's on my
                list?” to see <code>list_todos</code>.
            </p>
        </aside>
    );
}
