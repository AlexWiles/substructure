// Tools this skill unlocks once loaded.
export default [
    {
        name: "convert_units",
        description: "Convert a value between two units (km, mi, kg, lb).",
        input: {
            type: "object",
            properties: {
                value: { type: "number" },
                from: { type: "string" },
                to: { type: "string" }
            },
            required: ["value", "from", "to"]
        },
        exec: ({ value, from, to }) => {
            const rate = { km_mi: 0.621371, mi_km: 1.60934, kg_lb: 2.20462, lb_kg: 0.453592 };
            const k = `${from}_${to}`;
            return rate[k] ? `${value} ${from} = ${(value * rate[k]).toFixed(4)} ${to}` : `no rule for ${k}`;
        }
    }
];
