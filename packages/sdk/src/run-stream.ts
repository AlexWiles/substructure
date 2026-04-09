import type { Decimal, Event, TurnCompleted } from "./types";

export interface TurnResult {
    turnId: string;
    data: unknown;
    cost: Decimal;
    tokenUsage: Record<string, number>;
}

export class RunStream {
    readonly result: Promise<TurnResult>;
    private events: Event[] = [];
    private resolveResult!: (r: TurnResult) => void;
    private rejectResult!: (e: Error) => void;
    private source: AsyncGenerator<Event>;

    constructor(source: AsyncGenerator<Event>) {
        this.source = source;
        this.result = new Promise<TurnResult>((resolve, reject) => {
            this.resolveResult = resolve;
            this.rejectResult = reject;
        });
    }

    async *[Symbol.asyncIterator](): AsyncGenerator<Event> {
        try {
            for await (const event of this.source) {
                this.events.push(event);
                yield event;
            }
            const tc = this.events.findLast(
                (e): e is Event & { payload: TurnCompleted } =>
                    e.payload.type === "turn.completed",
            );
            if (tc) {
                this.resolveResult({
                    turnId: tc.payload.turn_id,
                    data: tc.payload.data,
                    cost: tc.payload.turn_cost ?? "0",
                    tokenUsage: tc.payload.turn_token_usage ?? {},
                });
            } else {
                this.rejectResult(new Error("stream ended without turn.completed"));
            }
        } catch (err) {
            this.rejectResult(err instanceof Error ? err : new Error(String(err)));
            throw err;
        }
    }
}
