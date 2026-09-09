import { Link } from "react-router-dom";

import type { SceneSummary, UploadResponse } from "../api/types";
import { parseAcquisitionDate, parsePlatform } from "../utils/scene";
import UploadControl from "./UploadControl";
import "./TopBar.css";

interface TopBarProps {
  scenes: SceneSummary[];
  selectedScene: SceneSummary | null;
  onSelectScene: (id: string) => void;
  maskOpacity: number;
  onMaskOpacityChange: (value: number) => void;
  onUploaded: (response: UploadResponse) => void;
}

export default function TopBar({
  scenes,
  selectedScene,
  onSelectScene,
  maskOpacity,
  onMaskOpacityChange,
  onUploaded,
}: TopBarProps) {
  const date = selectedScene ? parseAcquisitionDate(selectedScene.filename) : null;
  const platform = selectedScene ? parsePlatform(selectedScene.filename) : null;

  return (
    <header className="top-bar">
      <div className="top-bar__section">
        <Link to="/" className="top-bar__logo-link" title="Back to home">
          <span className="top-bar__wordmark">SATQUERY</span>
        </Link>
        <div className="top-bar__divider" />
        <label className="top-bar__field">
          <span className="label">SCENE</span>
          <select
            className="top-bar__select tabular"
            value={selectedScene?.id ?? ""}
            onChange={(e) => onSelectScene(e.target.value)}
          >
            {scenes.length === 0 && <option value="">loading…</option>}
            {scenes.map((scene) => (
              <option key={scene.id} value={scene.id}>
                {scene.id}
                {scene.source === "upload" ? ` · upload · ${scene.kind}` : ""}
              </option>
            ))}
          </select>
        </label>
        <UploadControl onUploaded={onUploaded} />
      </div>

      <div className="top-bar__section top-bar__section--telemetry">
        <TelemetryField label="GSD" value={selectedScene ? `${selectedScene.gsd_metres.toFixed(0)} m` : "—"} />
        <TelemetryField label="SENSOR" value={selectedScene?.sensor.toUpperCase() ?? "—"} />
        <TelemetryField label="PLATFORM" value={platform ?? (selectedScene?.source === "upload" ? "UPLOAD" : "—")} />
        <TelemetryField label="ACQUIRED" value={date ?? "—"} />

        {selectedScene?.kind !== "change" && selectedScene?.kind !== "cross_modal" && (
          <>
            <div className="top-bar__divider" />
            <label className="top-bar__field top-bar__field--slider">
              <span className="label">MASK</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={maskOpacity}
                onChange={(e) => onMaskOpacityChange(Number(e.target.value))}
                className="top-bar__slider"
                style={{ accentColor: "var(--signal-active)" }}
              />
              <span className="tabular top-bar__slider-value">{Math.round(maskOpacity * 100)}%</span>
            </label>
          </>
        )}
      </div>
    </header>
  );
}

function TelemetryField({ label, value }: { label: string; value: string }) {
  return (
    <div className="top-bar__field">
      <span className="label">{label}</span>
      <span className="tabular top-bar__value">{value}</span>
    </div>
  );
}
