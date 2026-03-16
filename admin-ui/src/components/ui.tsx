import type { ReactNode } from 'react'

// ── Typography ─────────────────────────────────────────────────────────────

export function Mono({ secondary, children }: { secondary?: boolean; children: ReactNode }) {
  const color = secondary ? 'text-[var(--color-text-secondary)]' : 'text-[var(--color-text)]'
  return <span className={`font-mono text-xs ${color}`}>{children}</span>
}

// ── Layout ─────────────────────────────────────────────────────────────────

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center gap-2 py-0.5 pl-3 min-w-0">
      <span className="shrink-0 font-mono text-xs text-[var(--color-text-secondary)] w-24">{label}</span>
      <span className="min-w-0 truncate font-mono text-xs text-[var(--color-text)]">{value}</span>
    </div>
  )
}

export function Panel({ children }: { children: ReactNode }) {
  return <div className="border border-[var(--color-text-secondary)]/25 py-1">{children}</div>
}

// ── Table ──────────────────────────────────────────────────────────────────

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto border border-[var(--color-text-secondary)]/25">
      <table className="w-full font-mono text-xs">{children}</table>
    </div>
  )
}

export function THead({ children }: { children: ReactNode }) {
  return (
    <thead>
      <tr className="border-b border-[var(--color-text-secondary)]/25">{children}</tr>
    </thead>
  )
}

export function Th({ children, align = 'left' }: { children: ReactNode; align?: 'left' | 'right' }) {
  return (
    <th className={`px-3 py-1.5 font-medium text-[var(--color-text-secondary)] text-${align}`}>
      {children}
    </th>
  )
}

export function TBody({ children }: { children: ReactNode }) {
  return <tbody className="divide-y divide-[var(--color-text-secondary)]/10">{children}</tbody>
}

export function Td({ children, secondary, align = 'left' }: { children: ReactNode; secondary?: boolean; align?: 'left' | 'right' }) {
  const color = secondary ? 'text-[var(--color-text-secondary)]' : 'text-[var(--color-text)]'
  return <td className={`px-3 py-1.5 ${color} text-${align}`}>{children}</td>
}

// ── Button ─────────────────────────────────────────────────────────────────

export function Button({ children, onClick, disabled }: { children: ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="border border-[var(--color-text-secondary)]/25 px-3 py-1 font-mono text-xs text-[var(--color-text-secondary)] transition hover:text-[var(--color-text)] disabled:opacity-50"
    >
      {children}
    </button>
  )
}
