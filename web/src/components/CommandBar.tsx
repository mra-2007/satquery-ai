import { useState } from "react";
import "./CommandBar.css";

export const ANALYZE_EXAMPLES = [
  "How many water bodies are there?",
  "How much forest is there in hectares?",
  "Is there any built-up area near water?",
  "How many buildings are there?",
  "Describe this scene",
  "Where is this?",
  "What's in this scene?",
];

export const CHANGE_EXAMPLES = [
  "What changed between these dates?",
  "Has built-up area increased?",
  "How much cropland is flooded?",
  "How much area is under water?",
  "Which built-up areas are near flooding?",
  "How much land changed to water?",
];

export const CROSS_MODAL_EXAMPLES = [
  "Does the SAR image confirm what the optical image shows?",
  "Compare the optical and SAR sensors.",
];

interface CommandBarProps {
  onSubmit: (query: string) => void;
  loading: boolean;
  examples?: string[];
}

export default function CommandBar({ onSubmit, loading, examples = ANALYZE_EXAMPLES }: CommandBarProps) {
  const [value, setValue] = useState("");

  function submit(query: string) {
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
  }

  return (
    <div className="command-bar">
      <form
        className="command-bar__form"
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
      >
        <span className="command-bar__prompt">›</span>
        <input
          className="command-bar__input"
          type="text"
          placeholder="Ask anything about this satellite scene…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={loading}
        />
        <button className="command-bar__submit" type="submit" disabled={loading || !value.trim()}>
          {loading ? "ANALYZING" : "ANALYZE"}
        </button>
      </form>
      <div className="command-bar__chips">
        {examples.map((example) => (
          <button
            key={example}
            className="command-bar__chip"
            type="button"
            disabled={loading}
            onClick={() => {
              setValue(example);
              submit(example);
            }}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
