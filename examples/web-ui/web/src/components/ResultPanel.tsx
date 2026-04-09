interface TurnResult {
    turnId: string;
    data: unknown;
    cost: string;
    tokenUsage: Record<string, number>;
}

interface ResultPanelProps {
    result: TurnResult;
    onReset: () => void;
}

export function ResultPanel({ result, onReset }: ResultPanelProps) {
    const totalTokens = Object.values(result.tokenUsage).reduce((sum, v) => sum + v, 0);

    return (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4 space-y-3">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-green-800">Research Complete</h3>
                <button
                    onClick={onReset}
                    className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                >
                    New Research
                </button>
            </div>
            <div className="flex gap-4 text-xs text-green-700">
                {result.cost !== "0" && <span>Cost: ${result.cost}</span>}
                {totalTokens > 0 && <span>Tokens: {totalTokens.toLocaleString()}</span>}
            </div>
        </div>
    );
}
