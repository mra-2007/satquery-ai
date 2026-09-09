import { useEffect, useState } from "react";
import type { TraceStep } from "../api/types";
import "./TraceDrawer.css";

interface TraceDrawerProps {
  trace: TraceStep[] | null;
  open: boolean;
  onClose: () => void;
}

/** A step is "geometry" -- deterministic measurement over the mask, per
 * CLAUDE.md's "No pixel, no claim" -- whenever its own recorded confidence
 * is exactly 1.0 (agent/dsl.py's execute() and agent/guardrail.py's
 * apply_capability_guardrail() both hardcode confidence=1.0 for exactly
 * this reason). Anything below 1.0 is the classifier's own uncertainty
 * (agent/tasks.py's classify_task) -- perception, not measurement. */
function isGeometryStep(step: TraceStep): boolean {
  return step.confidence >= 1;
}

export default function TraceDrawer({ trace, open, onClose }: TraceDrawerProps) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  // A fresh trace (new query) starts fully collapsed.
  useEffect(() => {
    setExpanded(new Set());
  }, [trace]);

  if (!open || !trace) return null;

  function toggleStep(index: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  return (
    <aside className="trace-drawer">
      <div className="trace-drawer__header">
        <div className="trace-drawer__title-block">
          <span className="label">Execution trace</span>
          <span className="tabular trace-drawer__step-count">{trace.length} steps</span>
        </div>
        <button className="trace-drawer__close" onClick={onClose} aria-label="Close execution trace">
          ×
        </button>
      </div>

      <p className="trace-drawer__caption">
        Uncertainty lives in perception, never in measurement -- every geometry step below
        reports exact confidence 1.00.
      </p>

      <div className="trace-drawer__legend">
        <span className="trace-drawer__legend-item">
          <span className="trace-drawer__legend-dot trace-drawer__legend-dot--geometry" />
          geometry -- deterministic
        </span>
        <span className="trace-drawer__legend-item">
          <span className="trace-drawer__legend-dot trace-drawer__legend-dot--perception" />
          perception -- classifier
        </span>
      </div>

      <ol className="trace-drawer__timeline">
        {trace.map((step, index) => {
          const geometry = isGeometryStep(step);
          const isExpanded = expanded.has(index);
          return (
            <li
              className={`trace-step ${geometry ? "trace-step--geometry" : "trace-step--perception"}`}
              key={index}
            >
              <div className="trace-step__rail">
                <span className="trace-step__dot" />
                <span className="trace-step__line" />
              </div>
              <div className="trace-step__body">
                <button
                  className="trace-step__summary"
                  onClick={() => toggleStep(index)}
                  aria-expanded={isExpanded}
                >
                  <div className="trace-step__head">
                    <span className="trace-step__task">{step.task}</span>
                    <span
                      className={
                        "trace-step__chevron" + (isExpanded ? " trace-step__chevron--open" : "")
                      }
                    >
                      ›
                    </span>
                  </div>
                  <span className="trace-step__tool tabular">{step.tool}</span>
                  <div className="trace-step__confidence-row">
                    <div className="trace-step__confidence-track">
                      <div
                        className={
                          "trace-step__confidence-fill" +
                          (geometry ? " trace-step__confidence-fill--geometry" : "")
                        }
                        style={{ width: `${step.confidence * 100}%` }}
                      />
                    </div>
                    <span className="tabular trace-step__confidence-value">
                      {step.confidence.toFixed(2)}
                    </span>
                  </div>
                </button>

                {isExpanded && (
                  <div className="trace-step__meta">
                    <div className="trace-step__meta-block">
                      <span className="label">Parameters</span>
                      <pre className="trace-step__json tabular">
                        {formatJson(step.parameters)}
                      </pre>
                    </div>
                    <div className="trace-step__meta-block">
                      <span className="label">Output</span>
                      <pre className="trace-step__json tabular">{formatJson(step.output)}</pre>
                    </div>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}

function formatJson(value: unknown): string {
  if (value === undefined) return "null";
  return JSON.stringify(value, null, 2);
}
