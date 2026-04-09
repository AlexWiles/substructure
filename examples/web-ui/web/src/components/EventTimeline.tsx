import { useEffect, useRef } from "react";
import type { Event } from "@substructure.ai/sdk";
import { EventCard } from "./EventCard";

const VISIBLE_TYPES = new Set([
    "turn.started",
    "llm.call.requested",
    "llm.call.completed",
    "llm.call.errored",
    "tool.call.requested",
    "tool.call.completed",
    "tool.call.errored",
    "message.new",
    "turn.completed",
]);

interface EventTimelineProps {
    events: Event[];
}

export function EventTimeline({ events }: EventTimelineProps) {
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [events.length]);

    const visible = events.filter((e) => VISIBLE_TYPES.has(e.payload.type));

    if (visible.length === 0) {
        return (
            <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
                Events will appear here as the agent works...
            </div>
        );
    }

    return (
        <div className="flex-1 overflow-y-auto space-y-2 py-2">
            {visible.map((event) => (
                <EventCard key={event.id} event={event} />
            ))}
            <div ref={endRef} />
        </div>
    );
}
