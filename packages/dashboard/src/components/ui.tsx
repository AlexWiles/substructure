import type { ReactNode } from "react";
import { Link } from "@tanstack/react-router";

// ── Typography ─────────────────────────────────────────────────────────────

export function Mono({ secondary, children }: { secondary?: boolean; children: ReactNode }) {
    const color = secondary ? "text-[var(--color-text-secondary)]" : "text-[var(--color-text)]";
    return <span className={`font-mono text-xs ${color}`}>{children}</span>;
}

export function SectionLabel({ children }: { children: ReactNode }) {
    return <h2 className="mb-3 font-mono text-xs font-semibold text-[var(--color-text)]">{children}</h2>;
}

export function Page({ children }: { children: ReactNode }) {
    return <main className="mx-auto max-w-5xl px-4 pb-8 pt-4">{children}</main>;
}

// ── Navigation ────────────────────────────────────────────────────────────

export interface Crumb {
    label: string;
    to: string;
    params?: Record<string, string>;
}

export function Breadcrumbs({ crumbs }: { crumbs: Crumb[] }) {
    return (
        <nav className="mb-4 flex items-center gap-1 font-mono text-xs">
            {crumbs.map((crumb, i) => {
                const isLast = i === crumbs.length - 1;
                return (
                    <span key={crumb.to} className="flex items-center gap-1">
                        {i > 0 && <span className="text-[var(--color-text-secondary)]">/</span>}
                        {isLast ? (
                            <span className="text-[var(--color-text)]">{crumb.label}</span>
                        ) : (
                            <Link
                                to={crumb.to}
                                params={crumb.params}
                                className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                            >
                                {crumb.label}
                            </Link>
                        )}
                    </span>
                );
            })}
        </nav>
    );
}

// ── Layout ─────────────────────────────────────────────────────────────────

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
    return (
        <div className="flex items-center gap-2 py-0.5 pl-3 min-w-0">
            <span className="shrink-0 font-mono text-xs text-[var(--color-text-secondary)] w-24">{label}</span>
            <span className="min-w-0 truncate font-mono text-xs text-[var(--color-text)]">{value}</span>
        </div>
    );
}

export function Panel({ children }: { children: ReactNode }) {
    return <div className="border border-[var(--color-text-secondary)]/25 py-1">{children}</div>;
}

// ── Table ──────────────────────────────────────────────────────────────────

export function Table({ children }: { children: ReactNode }) {
    return (
        <div className="overflow-x-auto border border-[var(--color-text-secondary)]/25">
            <table className="w-full font-mono text-xs">{children}</table>
        </div>
    );
}

export function THead({ children }: { children: ReactNode }) {
    return (
        <thead>
            <tr className="border-b border-[var(--color-text-secondary)]/25">{children}</tr>
        </thead>
    );
}

export function Th({ children, align = "left" }: { children: ReactNode; align?: "left" | "right" }) {
    return <th className={`px-3 py-1.5 font-medium text-[var(--color-text-secondary)] text-${align}`}>{children}</th>;
}

export function TBody({ children }: { children: ReactNode }) {
    return <tbody className="divide-y divide-[var(--color-text-secondary)]/10">{children}</tbody>;
}

export function Td({
    children,
    secondary,
    align = "left",
}: {
    children: ReactNode;
    secondary?: boolean;
    align?: "left" | "right";
}) {
    const color = secondary ? "text-[var(--color-text-secondary)]" : "text-[var(--color-text)]";
    return <td className={`px-3 py-1.5 ${color} text-${align}`}>{children}</td>;
}

// ── Button ─────────────────────────────────────────────────────────────────

export function Button({
    children,
    onClick,
    disabled,
}: {
    children: ReactNode;
    onClick?: () => void;
    disabled?: boolean;
}) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className="border border-[var(--color-text-secondary)]/25 px-3 py-1 font-mono text-xs text-[var(--color-text-secondary)] transition hover:text-[var(--color-text)] disabled:opacity-50"
        >
            {children}
        </button>
    );
}
