import type { SessionStatus } from "@substructure.ai/sdk/types";

export function StatusBadge({ status }: { status: SessionStatus }) {
    let label: string;
    let className: string;

    if (status === "idle") {
        label = "Idle";
        className = "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    } else if (status === "done") {
        label = "Done";
        className = "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    } else if (typeof status === "object" && "interrupted" in status) {
        label = "Interrupted";
        className = "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
    } else {
        label = String(status);
        className = "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
    }

    return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${className}`}>{label}</span>;
}
