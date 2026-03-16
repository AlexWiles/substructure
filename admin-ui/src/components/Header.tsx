import { Link } from '@tanstack/react-router'
import ThemeToggle from './ThemeToggle'

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--color-border)] bg-[var(--color-bg)]">
      <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
        <Link to="/" className="text-sm font-semibold text-[var(--color-text)] no-underline">
          Substructure
        </Link>
        <Link
          to="/sessions"
          className="text-sm text-[var(--color-text-secondary)] no-underline hover:text-[var(--color-text)]"
          activeProps={{ className: 'text-sm text-[var(--color-text)] font-medium no-underline' }}
        >
          Sessions
        </Link>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </nav>
    </header>
  )
}
