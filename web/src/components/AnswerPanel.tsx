import { useCountUp } from "../hooks/useCountUp";
import type { QueryResponse } from "../api/types";
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
}

export default function AnswerPanel({
  response,
  loading,
  error,
  traceOpen,
  onToggleTrace,
  verificationOpen,
  onToggleVerification,
}: AnswerPanelProps) {
  if (loading) {
    return (
      <aside className="answer-panel answer-panel--pending">
        <span className="label">MEASURING</span>
        <div className="answer-panel__pulse" />
      </aside>
    );
  }

  if (error) {
    return (
      <aside className="answer-panel answer-panel--error">
        <span className="label answer-panel__status-label answer-panel__status-label--error">
          QUERY FAILED
        </span>
        <p className="answer-panel__error-text">{error}</p>
      </aside>
    );
  }

  if (!response) {
    return (
      <aside className="answer-panel answer-panel--idle">
        <span className="label">AWAITING QUERY</span>
        <p className="answer-panel__idle-text">Ask a question about the scene to begin measuring.</p>
      </aside>
    );
  }

  const { evidence, limitation } = response;
  const isWithheld = limitation != null;
  const verification = findVerification(evidence.execution_trace);

  return (
    <aside
      className={`answer-panel ${isWithheld ? "answer-panel--withheld" : "answer-panel--verified"}`}
      key={response.report_id}
    >
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
          <span className="label answer-panel__status-label answer-panel__status-label--verified">
            Verified
          </span>
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
            <span className="tabular answer-panel__confidence-value">
              {entry.confidence.toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      {verification && (
        <>
          <hr className="hairline answer-panel__rule" />
          <button
            className="answer-panel__trace-toggle"
            onClick={onToggleVerification}
            aria-expanded={verificationOpen}
          >
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
              className={
                "answer-panel__trace-chevron" + (verificationOpen ? " answer-panel__trace-chevron--open" : "")
              }
            >
              ›
            </span>
          </button>
        </>
      )}

      <hr className="hairline answer-panel__rule" />
      <button
        className="answer-panel__trace-toggle"
        onClick={onToggleTrace}
        aria-expanded={traceOpen}
      >
        <span className="label">Execution trace</span>
        <span className="tabular answer-panel__trace-count">
          {evidence.execution_trace.length} steps
        </span>
        <span
          className={"answer-panel__trace-chevron" + (traceOpen ? " answer-panel__trace-chevron--open" : "")}
        >
          ›
        </span>
      </button>

      <p className="answer-panel__query">"{response.query}"</p>
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
