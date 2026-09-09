import type { LegendEntry } from "../api/types";
import "./Legend.css";

interface LegendProps {
  entries: LegendEntry[];
}

export default function Legend({ entries }: LegendProps) {
  if (entries.length === 0) return null;

  return (
    <div className="legend">
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
  );
}
