import { Link } from "@tanstack/react-router";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
    return (
        <header className="sticky top-0 z-50 border-b border-[var(--color-text-secondary)]/25 bg-[var(--color-surface)]">
            <nav className="mx-auto flex max-w-5xl items-center gap-4 px-4 py-2">
                <Link to="/" className="font-mono text-xs font-semibold text-[var(--color-text)] no-underline">
                    substructure.ai
                </Link>
                <Link
                    to="/sessions"
                    className="font-mono text-xs text-[var(--color-text-secondary)] no-underline hover:text-[var(--color-text)]"
                    activeProps={{ className: "font-mono text-xs text-[var(--color-text)] no-underline" }}
                >
                    sessions
                </Link>
                <div className="ml-auto">
                    <ThemeToggle />
                </div>
            </nav>
        </header>
    );
}
