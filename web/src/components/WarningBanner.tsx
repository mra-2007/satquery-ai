import "./WarningBanner.css";

interface WarningBannerProps {
  warnings: string[];
}

export default function WarningBanner({ warnings }: WarningBannerProps) {
  if (warnings.length === 0) return null;

  return (
    <div className="warning-banner">
      <span className="label warning-banner__label">REDUCED ACCURACY</span>
      <p className="warning-banner__text">{warnings.join("  ·  ")}</p>
    </div>
  );
}
