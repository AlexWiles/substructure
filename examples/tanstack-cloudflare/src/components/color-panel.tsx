import { useSyncExternalStore } from "react";

import { getColor, type Rgb, setColor, subscribeColor, toHex } from "../lib/color-store";

const CHANNELS: { key: keyof Rgb; label: string }[] = [
    { key: "r", label: "Red" },
    { key: "g", label: "Green" },
    { key: "b", label: "Blue" },
];

// The thing the agent drives. Its sliders + swatch are bound to the shared
// color store, so the frontend `set_color` tool (in any client) moves them — and
// you can drag them yourself too.
export function ColorPanel() {
    const color = useSyncExternalStore(subscribeColor, getColor, getColor);
    const hex = toHex(color);

    return (
        <aside className="colorpanel">
            <div className="colorpanel-title">Color mixer</div>
            <div className="swatch" style={{ background: hex }}>
                <span className="swatchhex">{hex.toUpperCase()}</span>
            </div>
            <div className="sliders">
                {CHANNELS.map(({ key, label }) => (
                    <label key={key} className="slider" data-ch={key}>
                        <span className="sliderlabel">{label}</span>
                        <input
                            type="range"
                            min={0}
                            max={255}
                            value={color[key]}
                            onChange={(e) => setColor({ [key]: Number(e.target.value) })}
                        />
                        <output className="sliderval">{color[key]}</output>
                    </label>
                ))}
            </div>
            <p className="panelhint">
                The agent's <code>set_color</code> moves these (“make it a calming teal”);
                drag them yourself and ask “what color is this?” to see <code>get_color</code>.
            </p>
        </aside>
    );
}
