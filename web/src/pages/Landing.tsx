import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { getScenes, sceneImageUrl, sceneMaskUrl } from "../api/client";
import { PREFERRED_DEFAULT_SCENE_ID } from "../utils/scene";
import { useCountUp } from "../hooks/useCountUp";
import "./Landing.css";

// --- Real, disclosed numbers only -- see eval/RESULTS.md and this
// session's own venv checks (pytest -q; onnx initializer sum). Nothing on
// this page is invented; every figure here traces back to a real script
// output or a real file on disk. -------------------------------------------

const GROUND_TRUTH_ACCURACY = 97.1; // eval/RESULTS.md, BigEarthNet.txt benchmark scored against the ground-truth mask
const PREDICTED_ACCURACY = 62.7; // same benchmark, scored against the real model's predicted mask
const RSVQA_ACCURACY = 39.5; // eval/RESULTS.md, RSVQA-LR (degraded-input stress test, see its own domain caveat)

const TEST_COUNT = 334; // venv\Scripts\python.exe -m pytest -q, this revision
const CLASS_COUNT = 19; // agent/vocabulary.py's SEGMENTATION_CLASSES
const PARAM_COUNT_M = 1.9; // sum of models/satquery_model.onnx's initializer tensor sizes, /1e6

export default function Landing() {
  return (
    <div className="landing">
      <Hero />
      <Problem />
      <HowItWorks />
      <Results />
      <CounterStrip />
      <Capabilities />
      <Footer />
    </div>
  );
}

// --- 1. HERO -----------------------------------------------------------------

function Hero() {
  const [sceneId, setSceneId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getScenes()
      .then((scenes) => {
        if (cancelled || scenes.length === 0) return;
        const preferred = scenes.find((s) => s.id === PREFERRED_DEFAULT_SCENE_ID);
        setSceneId((preferred ?? scenes[0]).id);
      })
      .catch(() => {
        // The hero degrades to a plain graphite background if the backend
        // isn't reachable -- never a broken image, never invented imagery.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="hero">
      <div className="hero__visual" aria-hidden="true">
        {sceneId && (
          <div className="hero__visual-drift">
            <img className="hero__image" src={sceneImageUrl(sceneId)} alt="" />
            <img className="hero__mask" src={sceneMaskUrl(sceneId)} alt="" />
          </div>
        )}
        <div className="hero__scrim" />
      </div>

      <div className="hero__content">
        <span className="label hero__eyebrow">No pixel, no claim</span>
        <h1 className="hero__title">SatQuery AI</h1>
        <p className="hero__tagline">Other systems generate answers. Ours computes them.</p>
        <Link to="/analyze" className="hero__cta">
          LAUNCH ANALYSIS
        </Link>
      </div>
    </section>
  );
}

// --- 2. THE PROBLEM ------------------------------------------------------------

const PROBLEM_CARDS = [
  {
    title: "Single-task silos",
    body: "One model counts. Another captions. Another flags change. A real question crosses all three, and nothing stitches them together.",
  },
  {
    title: "Expert knowledge required",
    body: "Reading a segmentation mask, judging what a sensor can and can't resolve, cross-checking optical against SAR -- normally demands a remote-sensing specialist at the console.",
  },
  {
    title: "Optical goes blind under cloud",
    body: "Cloud cover blinds every optical-only pipeline at exactly the moment radar keeps seeing straight through it.",
  },
];

function Problem() {
  return (
    <section className="section problem">
      <SectionHeading eyebrow="THE PROBLEM" title="Satellite AI answers questions no one asked" />
      <div className="problem__grid">
        {PROBLEM_CARDS.map((card) => (
          <div className="problem-card" key={card.title}>
            <h3 className="problem-card__title">{card.title}</h3>
            <p className="problem-card__body">{card.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

// --- 3. HOW IT WORKS -----------------------------------------------------------

const FLOW_NODES = ["QUESTION", "AI PLANS", "MODELS PERCEIVE", "GEOMETRY MEASURES", "VERIFIED ANSWER"];

function HowItWorks() {
  return (
    <section className="section how-it-works">
      <SectionHeading
        eyebrow="HOW IT WORKS"
        title="The model never answers. It only plans."
        subtitle="Every number is computed by deterministic geometry, then independently re-verified -- never taken on the language model's word."
      />
      <div className="flow">
        <div className="flow__track">
          <div className="flow__pulse" />
          {FLOW_NODES.map((label, i) => (
            <div className="flow__node" style={{ animationDelay: `${i * 1.2}s` }} key={label}>
              <span className="flow__node-dot" />
              <span className="flow__node-label">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// --- 4. RESULTS ------------------------------------------------------------------

function Results() {
  return (
    <section className="section results">
      <SectionHeading eyebrow="RESULTS" title="Every number below is a real script's real output" />
      <div className="results__primary">
        <div className="results__stat results__stat--huge">
          <StatNumber value={GROUND_TRUTH_ACCURACY} decimals={1} suffix="%" />
          <span className="results__stat-label">
            ground-truth accuracy
            <br />
            the reasoning layer alone, perfect segmentation assumed
          </span>
        </div>
        <div className="results__gap">
          <div className="results__stat results__stat--large">
            <StatNumber value={PREDICTED_ACCURACY} decimals={1} suffix="%" />
            <span className="results__stat-label">predicted accuracy, real model output</span>
          </div>
          <p className="results__gap-note">
            The 34-point gap is real segmentation error from the model itself -- isolated and measurable,
            not a flaw in the reasoning pipeline.
          </p>
        </div>
      </div>
      <div className="results__secondary">
        <StatNumber value={RSVQA_ACCURACY} decimals={1} suffix="%" className="results__stat--small" />
        <span className="results__secondary-label">
          on RSVQA-LR -- a harder, deliberately degraded-input stress test (13 of 16 model channels
          zero-filled), not a representative measurement.
        </span>
      </div>
    </section>
  );
}

// --- 5. COUNTER STRIP --------------------------------------------------------------

function CounterStrip() {
  return (
    <section className="counter-strip">
      <div className="counter-strip__item">
        <StatNumber value={TEST_COUNT} className="counter-strip__number" />
        <span className="counter-strip__label">tests passing</span>
      </div>
      <div className="counter-strip__divider" />
      <div className="counter-strip__item">
        <StatNumber value={CLASS_COUNT} className="counter-strip__number" />
        <span className="counter-strip__label">land-cover classes</span>
      </div>
      <div className="counter-strip__divider" />
      <div className="counter-strip__item">
        <StatNumber value={PARAM_COUNT_M} decimals={1} suffix="M" className="counter-strip__number" />
        <span className="counter-strip__label">parameters</span>
      </div>
      <div className="counter-strip__divider" />
      <div className="counter-strip__item">
        <span className="counter-strip__number tabular">0</span>
        <span className="counter-strip__label">GPUs required in the query path</span>
      </div>
    </section>
  );
}

// --- 6. CAPABILITIES ---------------------------------------------------------------

const CAPABILITIES = [
  { title: "Single-image VQA", body: "Count, measure, and locate land-cover -- computed, never guessed." },
  { title: "Captioning", body: "Plain-language scene descriptions, grounded in the mask's own facts." },
  { title: "Text-guided grounding", body: "Locate the exact region a natural-language phrase refers to." },
  { title: "Bi-temporal change", body: "Per-class area gained and lost between two real dates." },
  { title: "Optical-SAR fusion", body: "Cross-modal comparison that keeps seeing when optical goes blind." },
  { title: "Agentic orchestration", body: "The LLM plans tool calls. Geometry computes every answer." },
];

function Capabilities() {
  return (
    <section className="section capabilities">
      <SectionHeading eyebrow="CAPABILITIES" title="One agent, six real tools" />
      <div className="capabilities__grid">
        {CAPABILITIES.map((cap) => (
          <div className="capability-card" key={cap.title}>
            <h3 className="capability-card__title">{cap.title}</h3>
            <p className="capability-card__body">{cap.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

// --- 7. FOOTER -----------------------------------------------------------------------

function Footer() {
  return (
    <footer className="landing-footer">
      <span>Team Terrabyte</span>
      <span className="landing-footer__dot">·</span>
      <span>Smart India Hackathon 2026</span>
      <span className="landing-footer__dot">·</span>
      <span>PS SIH26167</span>
      <span className="landing-footer__dot">·</span>
      <span>ISRO / Space Applications Centre</span>
    </footer>
  );
}

// --- Shared: section heading, scroll-triggered count-up ------------------------------

function SectionHeading({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle?: string }) {
  return (
    <div className="section-heading">
      <span className="label section-heading__eyebrow">{eyebrow}</span>
      <h2 className="section-heading__title">{title}</h2>
      {subtitle && <p className="section-heading__subtitle">{subtitle}</p>}
    </div>
  );
}

/** Reveals a numeric target and animates from 0 to it only once the
 * element scrolls into view -- the hero and everything above the fold
 * loads instantly, but a stat below the fold draws the eye by counting up
 * exactly when it's first seen, the same way AnswerPanel's own
 * useCountUp draws the eye to a fresh answer. */
function StatNumber({
  value,
  decimals = 0,
  suffix = "",
  className = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const animated = useCountUp(inView ? value : 0, 1400);
  const display = typeof animated === "number" ? animated.toFixed(decimals) : animated;

  return (
    <span ref={ref} className={`tabular ${className}`}>
      {display}
      {suffix}
    </span>
  );
}
