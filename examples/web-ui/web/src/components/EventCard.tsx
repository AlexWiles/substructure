import type { Event } from "@substructure.ai/sdk";

function truncate(s: string, max: number): string {
    return s.length > max ? s.slice(0, max) + "..." : s;
}

function formatContent(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content
            .filter((p): p is { type: "text"; text: string } => p.type === "text")
            .map((p) => p.text)
            .join("\n");
    }
    return "";
}

interface EventCardProps {
    event: Event;
}

export function EventCard({ event }: EventCardProps) {
    const p = event.payload;

    switch (p.type) {
        case "turn.started":
            return (
                <div className="flex items-center gap-2 text-xs text-gray-400 py-1">
                    <div className="h-px flex-1 bg-gray-200" />
                    <span>Turn started</span>
                    <div className="h-px flex-1 bg-gray-200" />
                </div>
            );

        case "llm.call.requested":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-purple-50 border border-purple-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-purple-100">
                        <span className="animate-spin text-sm">~</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-purple-700">
                            Thinking...
                        </p>
                        <p className="text-xs text-purple-500">
                            {p.request.model}
                        </p>
                    </div>
                </div>
            );

        case "llm.call.completed":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-purple-50 border border-purple-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-purple-200 text-purple-700">
                        <span className="text-sm">&#10003;</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-purple-700">
                            LLM response received
                        </p>
                        <p className="text-xs text-purple-500">
                            {p.response.model} &middot; {p.response.finish_reason}
                        </p>
                    </div>
                </div>
            );

        case "llm.call.errored":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-red-50 border border-red-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-red-200 text-red-700">
                        <span className="text-sm">!</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-red-700">
                            LLM Error
                        </p>
                        <p className="text-xs text-red-500">
                            {truncate(p.error, 200)}
                        </p>
                    </div>
                </div>
            );

        case "tool.call.requested":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-blue-50 border border-blue-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                        <span className="text-sm">&#9881;</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-blue-700">
                            {p.name}
                        </p>
                        <p className="text-xs text-blue-500 font-mono break-all">
                            {truncate(p.arguments, 150)}
                        </p>
                    </div>
                </div>
            );

        case "tool.call.completed":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-green-50 border border-green-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-green-200 text-green-700">
                        <span className="text-sm">&#10003;</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-green-700">
                            {p.name}
                        </p>
                        <p className="text-xs text-green-600 font-mono break-all">
                            {truncate(p.result, 200)}
                        </p>
                    </div>
                </div>
            );

        case "tool.call.errored":
            return (
                <div className="flex items-center gap-3 rounded-lg bg-red-50 border border-red-100 px-4 py-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-red-200 text-red-700">
                        <span className="text-sm">&#10007;</span>
                    </div>
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-red-700">
                            {p.name} failed
                        </p>
                        <p className="text-xs text-red-500">
                            {truncate(p.error, 200)}
                        </p>
                    </div>
                </div>
            );

        case "message.new": {
            const content = formatContent(p.message.content);
            if (!content) return null;

            const isUser = p.message.role === "user";
            return (
                <div
                    className={`rounded-lg border px-4 py-3 ${
                        isUser
                            ? "bg-gray-50 border-gray-200"
                            : "bg-white border-gray-200"
                    }`}
                >
                    <p
                        className={`text-xs font-semibold mb-1 ${
                            isUser ? "text-gray-500" : "text-gray-700"
                        }`}
                    >
                        {isUser ? "User" : "Assistant"}
                    </p>
                    <p className="text-sm text-gray-800 whitespace-pre-wrap">
                        {content}
                    </p>
                </div>
            );
        }

        case "turn.completed":
            return (
                <div className="flex items-center gap-2 text-xs text-gray-400 py-1">
                    <div className="h-px flex-1 bg-gray-200" />
                    <span>
                        Turn complete
                        {p.turn_cost ? ` · $${p.turn_cost}` : ""}
                    </span>
                    <div className="h-px flex-1 bg-gray-200" />
                </div>
            );

        default:
            return null;
    }
}
