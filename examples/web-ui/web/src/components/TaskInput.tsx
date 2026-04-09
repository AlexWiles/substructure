import { useState } from "react";

const SUGGESTIONS = [
    "Quantum Computing",
    "Rust Programming Language",
    "Climate Change Solutions",
];

interface TaskInputProps {
    onSubmit: (topic: string) => void;
    disabled: boolean;
}

export function TaskInput({ onSubmit, disabled }: TaskInputProps) {
    const [value, setValue] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const trimmed = value.trim();
        if (!trimmed || disabled) return;
        onSubmit(trimmed);
        setValue("");
    };

    return (
        <div className="space-y-3">
            <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                    type="text"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    placeholder="Enter a research topic..."
                    disabled={disabled}
                    className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50 disabled:bg-gray-100"
                />
                <button
                    type="submit"
                    disabled={disabled || !value.trim()}
                    className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Research
                </button>
            </form>
            <div className="flex gap-2">
                <span className="text-xs text-gray-400 py-1">Try:</span>
                {SUGGESTIONS.map((s) => (
                    <button
                        key={s}
                        onClick={() => {
                            if (!disabled) onSubmit(s);
                        }}
                        disabled={disabled}
                        className="rounded-full border border-gray-200 px-3 py-1 text-xs text-gray-600 hover:bg-gray-100 hover:border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {s}
                    </button>
                ))}
            </div>
        </div>
    );
}
