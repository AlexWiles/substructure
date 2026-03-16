import { memo, useState, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'

// ── Design system ──────────────────────────────────────────────────────────
// Depth is communicated through indent alone.
// No nested cards, no borders — every row lives in the same flat list.

const INDENT_PX = 20

// ── TreeRow ────────────────────────────────────────────────────────────────

export interface TreeRowProps {
  depth?: number
  icon: ReactNode
  label: ReactNode
  summary?: ReactNode
  trailing?: ReactNode
  children?: ReactNode
  defaultExpanded?: boolean
}

export const TreeRow = memo(function TreeRow({
  depth = 0,
  icon,
  label,
  summary,
  trailing,
  children,
  defaultExpanded = false,
}: TreeRowProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const hasChildren = children != null

  return (
    <>
      <button
        onClick={() => hasChildren && setExpanded(!expanded)}
        className={`flex w-full items-center gap-2 text-left transition-colors px-3 py-1 min-w-0 ${
          hasChildren ? 'hover:bg-[var(--color-hover)] cursor-pointer' : 'cursor-default'
        }`}
        style={{ paddingLeft: `${12 + depth * INDENT_PX}px` }}
      >
        <span className="flex w-3 shrink-0 items-center justify-center">
          {hasChildren ? (
            <ChevronRight
              size={12}
              className={`shrink-0 transition-transform text-[var(--color-text-secondary)] ${expanded ? 'rotate-90' : ''}`}
            />
          ) : (
            <span className="w-3" />
          )}
        </span>
        <span className="flex w-4 shrink-0 items-center justify-center">{icon}</span>
        <span className="shrink-0 font-mono text-xs font-medium text-[var(--color-text)] truncate max-w-40">
          {label}
        </span>
        {summary && (
          <span className="min-w-0 truncate text-xs text-[var(--color-text-secondary)]">
            {summary}
          </span>
        )}
        <span className="flex-1" />
        {trailing && (
          <span className="flex items-center gap-2 shrink-0">{trailing}</span>
        )}
      </button>
      {expanded && children}
    </>
  )
})
