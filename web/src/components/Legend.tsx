import type { LegendEntry } from "../api/types";
import "./Legend.css";

interface LegendProps {
  entries: LegendEntry[];
}

/** Collapsed by default to a compact, low-contrast row of colour chips --
 * expanding into the full class-name list only on hover (pure CSS, no
 * open/close state), so it stays out of the way of the map until wanted. */
export default function Legend({ entries }: LegendProps) {
  if (entries.length === 0) return null;

  return (
    <div className="legend">
      <div className="legend__chip-row">
        {entries.map((entry) => (
          <span
            key={entry.class_id}
            className="legend__chip"
            style={{ background: entry.color }}
            title={entry.class_name}
          />
        ))}
      </div>

      <div className="legend__expanded">
        <span className="label legend__title">CLASSES PRESENT</span>
        <div className="legend__rows">
          {entries.map((entry) => (
            <div className="legend__row" key={entry.class_id}>
              <span className="legend__swatch" style={{ background: entry.color }} />
              <span className="legend__name">{entry.class_name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
