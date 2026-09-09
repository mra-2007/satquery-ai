import { useState } from "react";

import { reportUrl } from "../api/client";
import { useCountUp } from "../hooks/useCountUp";
import type { QueryResponse } from "../api/types";
import { downloadJson } from "../utils/download";
import { findVerification } from "./VerificationPanel";
import "./AnswerPanel.css";

interface AnswerPanelProps {
  response: QueryResponse | null;
  loading: boolean;
  error: string | null;
  traceOpen: boolean;
  onToggleTrace: () => void;
  verificationOpen: boolean;
  onToggleVerification: () => void;
  // Narrower, smaller-type variant for the CROSS-MODAL view, where the
  // swipe comparison itself is the content and every panel has to give
  // most of the width back to the imagery -- see CrossModalView.tsx.
  // Adds its own collapse toggle so the panel can shrink to a thin strip.
  compact?: boolean;
}

export default function AnswerPanel({
  response,
  loading,
  error,
  traceOpen,
  onToggleTrace,
  verificationOpen,
  onToggleVerification,
  compact = false,
}: AnswerPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  let stateClass = "answer-panel--idle";
  let body: React.ReactNode = (
    <>
      <span className="label">AWAITING QUERY</span>
      <p className="answer-panel__idle-text">Ask a question about the scene to begin measuring.</p>
    </>
  );

  if (loading) {
    stateClass = "answer-panel--pending";
    body = (
      <>
        <span className="label">MEASURING</span>
        <div className="answer-panel__pulse" />
      </>
    );
  } else if (error) {
    stateClass = "answer-panel--error";
    body = (
      <>
        <span className="label answer-panel__status-label answer-panel__status-label--error">QUERY FAILED</span>
        <p className="answer-panel__error-text">{error}</p>
      </>
    );
  } else if (response) {
    const { evidence, limitation } = response;
    const isWithheld = limitation != null;
    const verification = findVerification(evidence.execution_trace);
    stateClass = isWithheld ? "answer-panel--withheld" : "answer-panel--verified";

    body = (
      <>
        {isWithheld ? (
          <>
            <span className="label answer-panel__status-label answer-panel__status-label--withheld">
              Exact answer withheld
            </span>
            <hr className="hairline answer-panel__rule" />
            <div className="answer-panel__reason-block">
              <span className="label">WHY</span>
              <p className="answer-panel__reason-text">{limitation}</p>
            </div>
            <hr className="hairline answer-panel__rule" />
            <div className="answer-panel__reason-block">
              <span className="label">COMPUTED INSTEAD</span>
              <AnswerValue value={evidence.value} units={evidence.units} tone="withheld" />
            </div>
          </>
        ) : (
          <>
            <span className="label answer-panel__status-label answer-panel__status-label--verified">Verified</span>
            <AnswerValue value={evidence.value} units={evidence.units} tone="verified" />
          </>
        )}

        <hr className="hairline answer-panel__rule" />
        <div className="answer-panel__confidence">
          <span className="label">CONFIDENCE</span>
          {evidence.confidence.map((entry) => (
            <div className="answer-panel__confidence-row" key={entry.source}>
              <span className="answer-panel__confidence-source">{entry.source}</span>
              <div className="answer-panel__confidence-bar-track">
                <div
                  className={
                    "answer-panel__confidence-bar-fill" +
                    (entry.confidence >= 1 ? " answer-panel__confidence-bar-fill--exact" : "")
                  }
                  style={{ width: `${entry.confidence * 100}%` }}
                />
              </div>
              <span className="tabular answer-panel__confidence-value">{entry.confidence.toFixed(2)}</span>
            </div>
          ))}
        </div>

        {verification && (
          <>
            <hr className="hairline answer-panel__rule" />
            <button className="answer-panel__trace-toggle" onClick={onToggleVerification} aria-expanded={verificationOpen}>
              <span className="label">Verification</span>
              <span
                className={
                  "tabular answer-panel__trace-count" +
                  (verification.all_passed
                    ? " answer-panel__verification-count--pass"
                    : " answer-panel__verification-count--fail")
                }
              >
                {verification.claims.filter((c) => c.passed).length}/{verification.claims.length} claims
              </span>
              <span
                className={"answer-panel__trace-chevron" + (verificationOpen ? " answer-panel__trace-chevron--open" : "")}
              >
                ›
              </span>
            </button>
          </>
        )}

        <hr className="hairline answer-panel__rule" />
        <button className="answer-panel__trace-toggle" onClick={onToggleTrace} aria-expanded={traceOpen}>
          <span className="label">Execution trace</span>
          <span className="tabular answer-panel__trace-count">{evidence.execution_trace.length} steps</span>
          <span className={"answer-panel__trace-chevron" + (traceOpen ? " answer-panel__trace-chevron--open" : "")}>
            ›
          </span>
        </button>

        <hr className="hairline answer-panel__rule" />
        <div className="answer-panel__export-row">
          <a
            className="answer-panel__export-link"
            href={reportUrl(response.report_id)}
            download={`report_${response.report_id}.json`}
          >
            <span aria-hidden="true">⭳</span> REPORT
          </a>
          <button
            type="button"
            className="answer-panel__export-link"
            onClick={() => downloadJson(`geometry_${response.report_id}.geojson`, evidence.geometry)}
          >
            <span aria-hidden="true">⭳</span> GEOJSON
          </button>
        </div>

        <p className="answer-panel__query">"{response.query}"</p>
      </>
    );
  }

  const panelClass =
    "answer-panel " +
    stateClass +
    (compact ? " answer-panel--compact" : "") +
    (compact && collapsed ? " answer-panel--collapsed" : "");

  return (
    <aside className={panelClass} key={response?.report_id}>
      {compact && (
        <button
          type="button"
          className="answer-panel__collapse-toggle"
          onClick={() => setCollapsed((v) => !v)}
          aria-label={collapsed ? "Expand answer panel" : "Collapse answer panel"}
        >
          {collapsed ? "‹" : "›"}
        </button>
      )}
      {!(compact && collapsed) && body}
    </aside>
  );
}

function AnswerValue({
  value,
  units,
  tone,
}: {
  value: boolean | number | string;
  units: string | null;
  tone: "verified" | "withheld";
}) {
  if (typeof value === "boolean") {
    return (
      <div className={`answer-panel__value-block answer-panel__value-block--${tone}`}>
        <span className={`answer-panel__number tabular answer-panel__number--${tone}`}>
          {value ? "YES" : "NO"}
        </span>
      </div>
    );
  }

  if (typeof value === "number") {
    return <NumericAnswer value={value} units={units} tone={tone} />;
  }

  return (
    <div className={`answer-panel__value-block answer-panel__value-block--${tone}`}>
      <p className="answer-panel__text-answer">{value}</p>
      {units && <span className="answer-panel__units">{units}</span>}
    </div>
  );
}

function NumericAnswer({
  value,
  units,
  tone,
}: {
  value: number;
  units: string | null;
  tone: "verified" | "withheld";
}) {
  const animated = useCountUp(value);
  const display = typeof animated === "number" ? formatNumber(animated) : animated;

  return (
    <div className={`answer-panel__value-block answer-panel__value-block--${tone}`}>
      <span className={`answer-panel__number tabular answer-panel__number--${tone}`}>{display}</span>
      {units && <span className="answer-panel__units">{units}</span>}
    </div>
  );
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}
