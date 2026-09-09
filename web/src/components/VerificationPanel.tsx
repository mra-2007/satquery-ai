import type { FusionSourceEvidence, FusionSummary, TraceStep, VerificationSummary } from "../api/types";
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

/** Finds the one trace step evidence/fusion.py's fuse_evidence() produced,
 * whenever a plan explicitly called the "fusion" tool -- identifiable
 * purely by tool name, same as findVerification above. */
export function findFusion(trace: TraceStep[] | null): FusionSummary | null {
  if (!trace) return null;
  const step = trace.find((s) => s.tool === "fusion");
  return (step?.output as FusionSummary | undefined) ?? null;
}

export default function VerificationPanel({ trace, open, onClose }: VerificationPanelProps) {
  const verification = findVerification(trace);
  const fusion = findFusion(trace);
  if (!open || (!verification && !fusion)) return null;

  const passedCount = verification?.claims.filter((c) => c.passed).length ?? 0;
  const totalCount = verification?.claims.length ?? 0;

  return (
    <aside className="verification-panel">
      <div className="verification-panel__header">
        <div className="verification-panel__title-block">
          <span className="label">Verification</span>
          {verification && (
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
          )}
        </div>
        <button className="verification-panel__close" onClick={onClose} aria-label="Close verification">
          ×
        </button>
      </div>

      {verification && (
        <>
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
        </>
      )}

      {fusion && <FusionSection fusion={fusion} />}
    </aside>
  );
}

function FusionSection({ fusion }: { fusion: FusionSummary }) {
  const availableCount = fusion.sources.filter((s) => s.available).length;

  return (
    <div className="fusion-section">
      <div className="fusion-section__header">
        <span className="label">Cross-source fusion</span>
        <span
          className={
            "tabular fusion-section__verdict" +
            (fusion.fused_present ? " fusion-section__verdict--present" : " fusion-section__verdict--absent")
          }
        >
          {fusion.fused_present ? "PRESENT" : "ABSENT"} · {(fusion.fused_confidence * 100).toFixed(0)}%
        </span>
      </div>

      <p className="verification-panel__caption">
        {fusion.class_name} -- {availableCount}/{fusion.sources.length} sources available,{" "}
        {(fusion.agreement_fraction * 100).toFixed(0)}% agree with the fused verdict.
      </p>

      <p className="fusion-section__weighting-note">{fusion.weighting_note}</p>

      <ol className="fusion-section__sources">
        {fusion.sources.map((source) => (
          <FusionSourceRow key={source.source} source={source} correlated={fusion.correlated_sources.includes(source.source)} />
        ))}
      </ol>

      <p className="fusion-section__summary">{fusion.summary}</p>
    </div>
  );
}

function FusionSourceRow({ source, correlated }: { source: FusionSourceEvidence; correlated: boolean }) {
  const verdictLabel = !source.available ? "N/A" : source.present ? "PRESENT" : "ABSENT";

  return (
    <li
      className={
        "fusion-source" +
        (!source.available
          ? " fusion-source--unavailable"
          : source.present
            ? " fusion-source--present"
            : " fusion-source--absent")
      }
    >
      <div className="fusion-source__head">
        <span className="fusion-source__name">
          {source.source.replace("_", " ")}
          {correlated && <span className="fusion-source__correlated-tag">correlated</span>}
        </span>
        <span className="tabular fusion-source__weight">×{source.weight.toFixed(1)}</span>
        <span className="tabular fusion-source__verdict">{verdictLabel}</span>
      </div>
      {source.available && (
        <div className="fusion-source__confidence-track">
          <div
            className="fusion-source__confidence-fill"
            style={{ width: `${(source.confidence ?? 0) * 100}%` }}
          />
        </div>
      )}
      <p className="fusion-source__detail">{source.detail}</p>
    </li>
  );
}

function formatValue(value: boolean | number | null): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}
