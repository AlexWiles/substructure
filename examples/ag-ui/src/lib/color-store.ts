// A tiny framework-agnostic store for the demo's color mixer. The frontend
// `set_color` tool — defined in each chat client — writes to it; the ColorPanel
// reads it via useSyncExternalStore. It's a module-global singleton, which is
// all a single-user demo needs and lets the (module-scope) tool executors reach
// the same state the panel renders.

export type Rgb = { r: number; g: number; b: number };

let state: Rgb = { r: 99, g: 102, b: 241 }; // indigo, to start
const listeners = new Set<() => void>();

// Animation bookkeeping. A monotonically-bumped token cancels any in-flight
// tween the moment a newer setColor() arrives (or a drag snaps the value).
let rafId = 0;
let animToken = 0;
const TWEEN_MS = 550;

export function getColor(): Rgb {
    return state;
}

/**
 * Apply a (partial) color and notify subscribers; returns the clamped target.
 *
 * With `{ animate: true }` (used by the agent's tool) it tweens from the current
 * color to the target over a few hundred ms, so the sliders glide and the swatch
 * morphs. Manual slider drags omit it and snap instantly.
 */
export function setColor(next: Partial<Rgb>, opts: { animate?: boolean } = {}): Rgb {
    const target: Rgb = {
        r: clamp(next.r ?? state.r),
        g: clamp(next.g ?? state.g),
        b: clamp(next.b ?? state.b),
    };

    cancelTween();

    // Snap when not animating, or when rAF isn't available (e.g. SSR).
    if (!opts.animate || typeof requestAnimationFrame === "undefined") {
        state = target;
        emit();
        return target;
    }

    const from = state;
    const token = ++animToken;
    let startTs = -1;
    const step = (ts: number) => {
        if (token !== animToken) return; // superseded
        if (startTs < 0) startTs = ts;
        const p = Math.min(1, (ts - startTs) / TWEEN_MS);
        const e = easeOutCubic(p);
        state = {
            r: lerp(from.r, target.r, e),
            g: lerp(from.g, target.g, e),
            b: lerp(from.b, target.b, e),
        };
        emit();
        if (p < 1) rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);

    // The tool result should carry the final color, not a mid-tween frame.
    return target;
}

function emit(): void {
    for (const notify of listeners) notify();
}

function cancelTween(): void {
    animToken++;
    if (rafId && typeof cancelAnimationFrame !== "undefined") cancelAnimationFrame(rafId);
    rafId = 0;
}

function lerp(a: number, b: number, t: number): number {
    return Math.round(a + (b - a) * t);
}

function easeOutCubic(t: number): number {
    return 1 - (1 - t) ** 3;
}

export function subscribeColor(onChange: () => void): () => void {
    listeners.add(onChange);
    return () => {
        listeners.delete(onChange);
    };
}

export function toHex({ r, g, b }: Rgb): string {
    return "#" + [r, g, b].map((n) => n.toString(16).padStart(2, "0")).join("");
}

function clamp(n: number): number {
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(255, Math.round(n)));
}
