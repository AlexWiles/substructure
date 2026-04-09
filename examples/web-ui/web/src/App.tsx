import { useAgent } from "./hooks/useAgent";
import { TaskInput } from "./components/TaskInput";
import { EventTimeline } from "./components/EventTimeline";
import { ResultPanel } from "./components/ResultPanel";

export function App() {
    const { status, events, result, error, submit, reset } = useAgent();

    return (
        <div className="mx-auto flex h-screen max-w-2xl flex-col px-4 py-8">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-xl font-bold text-gray-900">Research Agent</h1>
                <p className="text-sm text-gray-500">
                    Enter a topic and watch the agent search, read, and summarize — powered by{" "}
                    <span className="font-medium">Substructure</span> +{" "}
                    <span className="font-medium">FrontendClient</span>
                </p>
            </div>

            {/* Input */}
            <TaskInput onSubmit={submit} disabled={status === "running"} />

            {/* Status */}
            {status === "running" && (
                <div className="mt-4 flex items-center gap-2 text-sm text-blue-600">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500" />
                    Researching...
                </div>
            )}

            {error && (
                <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {error}
                </div>
            )}

            {/* Event Timeline */}
            <div className="mt-4 flex flex-1 flex-col overflow-hidden rounded-lg border border-gray-200 bg-white p-4">
                <EventTimeline events={events} />
            </div>

            {/* Result */}
            {result && (
                <div className="mt-4">
                    <ResultPanel result={result} onReset={reset} />
                </div>
            )}
        </div>
    );
}
