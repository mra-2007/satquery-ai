import type { TraceStep, VerificationSummary } from "../api/types";
import "./VerificationPanel.css";

interface VerificationPanelProps {
  trace: TraceStep[] | null;
  open: boolean;
  onClose: () => void;
}

/** Finds the one trace step tools/verifier.py's verify_answer() produced --
 * automatically appended by agent/executor.py after any caption step, or
 * present as an ordinary "execute:" step if a plan explicitly called the
 * "verify" tool. Either way it's identifiable purely by tool name. */
export function findVerification(trace: TraceStep[] | null): VerificationSummary | null {
  if (!trace) return null;
  const step = trace.find((s) => s.tool === "verify");
  return (step?.output as VerificationSummary | undefined) ?? null;
}

export default function VerificationPanel({ trace, open, onClose }: VerificationPanelProps) {
  const verification = findVerification(trace);
  if (!open || !verification) return null;

  const passedCount = verification.claims.filter((c) => c.passed).length;
  const totalCount = verification.claims.length;

  return (
    <aside className="verification-panel">
      <div className="verification-panel__header">
        <div className="verification-panel__title-block">
          <span className="label">Verification</span>
          <span
            className={
              "tabular verification-panel__status" +
              (verification.all_passed
                ? " verification-panel__status--pass"
                : " verification-panel__status--fail")
            }
          >
            {passedCount}/{totalCount} claims verified
          </span>
        </div>
        <button className="verification-panel__close" onClick={onClose} aria-label="Close verification">
          ×
        </button>
      </div>

      <p className="verification-panel__caption">
        Every claim below was independently re-derived from the mask via evidence/ops.py --
        never taken on the generated text's word.
      </p>

      {totalCount === 0 ? (
        <p className="verification-panel__status-text">
          No checkable factual claims were found in this answer.
        </p>
      ) : (
        <ol className="verification-panel__claims">
          {verification.claims.map((claim, index) => (
            <li
              className={
                "verification-claim" +
                (claim.passed ? " verification-claim--pass" : " verification-claim--fail")
              }
              key={index}
            >
              <div className="verification-claim__head">
                <span className="verification-claim__badge">{claim.passed ? "✓" : "✗"}</span>
                <span className="verification-claim__text">"{claim.text}"</span>
              </div>
              <div className="verification-claim__meta">
                <span className="verification-claim__tool tabular">{claim.tool}</span>
                {claim.class_name && (
                  <span className="verification-claim__class">{claim.class_name}</span>
                )}
              </div>
              <div className="verification-claim__values tabular">
                <span>claimed: {formatValue(claim.claimed_value)}</span>
                <span>recomputed: {formatValue(claim.recomputed_value)}</span>
              </div>
              {claim.reason && <p className="verification-claim__reason">{claim.reason}</p>}
            </li>
          ))}
        </ol>
      )}

      {verification.verified_answer !== verification.original_answer && (
        <div className="verification-panel__answer-block">
          <span className="label">Verified answer</span>
          <p className="verification-panel__verified-text">{verification.verified_answer}</p>
        </div>
      )}
    </aside>
  );
}

function formatValue(value: boolean | number | null): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}
