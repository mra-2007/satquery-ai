import type { ChangeSummary } from "../api/types";
import "./GainLossTable.css";

interface GainLossTableProps {
  summary: ChangeSummary | null;
  loading: boolean;
  error: string | null;
}

export default function GainLossTable({ summary, loading, error }: GainLossTableProps) {
  if (loading) {
    return (
      <div className="gain-loss-table">
        <span className="label">PER-CLASS CHANGE</span>
        <p className="gain-loss-table__status">Computing change...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="gain-loss-table">
        <span className="label">PER-CLASS CHANGE</span>
        <p className="gain-loss-table__status gain-loss-table__status--error">{error}</p>
      </div>
    );
  }

  if (!summary) return null;

  return (
    <div className="gain-loss-table">
      <div className="gain-loss-table__header">
        <span className="label">PER-CLASS CHANGE</span>
        <span className="tabular gain-loss-table__changed-fraction">
          {(summary.changed_fraction * 100).toFixed(1)}% of pixels changed
        </span>
      </div>

      {summary.classes.length === 0 ? (
        <p className="gain-loss-table__status">No class-level change detected between these dates.</p>
      ) : (
        <>
          <div className="gain-loss-table__legend">
            <span className="gain-loss-table__legend-item">
              <span className="gain-loss-table__swatch gain-loss-table__swatch--gain" /> gained
            </span>
            <span className="gain-loss-table__legend-item">
              <span className="gain-loss-table__swatch gain-loss-table__swatch--loss" /> lost
            </span>
          </div>

          <div className="gain-loss-table__rows">
            <div className="gain-loss-table__row gain-loss-table__row--head">
              <span className="label">CLASS</span>
              <span className="label gain-loss-table__num-head">GAIN (HA)</span>
              <span className="label gain-loss-table__num-head">LOSS (HA)</span>
              <span className="label gain-loss-table__num-head">NET (HA)</span>
            </div>
            {summary.classes.map((entry) => (
              <div className="gain-loss-table__row" key={entry.class_id}>
                <span className="gain-loss-table__name">{entry.class_name}</span>
                <span className="tabular gain-loss-table__num gain-loss-table__num--gain">
                  {entry.gained_ha > 0 ? `+${entry.gained_ha.toFixed(2)}` : "–"}
                </span>
                <span className="tabular gain-loss-table__num gain-loss-table__num--loss">
                  {entry.lost_ha > 0 ? `−${entry.lost_ha.toFixed(2)}` : "–"}
                </span>
                <span
                  className={
                    "tabular gain-loss-table__num gain-loss-table__num--net " +
                    (entry.net_ha > 0
                      ? "gain-loss-table__num--net-positive"
                      : entry.net_ha < 0
                        ? "gain-loss-table__num--net-negative"
                        : "")
                  }
                >
                  {entry.net_ha > 0 ? "+" : ""}
                  {entry.net_ha.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      <p className="gain-loss-table__summary">{summary.summary}</p>
    </div>
  );
}
