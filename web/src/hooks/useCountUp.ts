import { useEffect, useState } from "react";

/**
 * Animates from the previous numeric value to `target` over `duration`ms.
 * Purposeful, not decorative: it draws the eye to the one number that
 * matters the instant a new answer arrives. Non-numeric targets (string
 * answers from caption/ground/etc.) are returned unchanged, immediately.
 */
export function useCountUp(target: number | string, duration = 220): number | string {
  const [display, setDisplay] = useState<number | string>(target);

  useEffect(() => {
    if (typeof target !== "number" || !Number.isFinite(target)) {
      setDisplay(target);
      return;
    }

    const start = performance.now();
    const from = typeof display === "number" ? display : 0;
    let frame: number;

    function tick(now: number) {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value = from + (target as number - from) * eased;
      setDisplay(Number.isInteger(target) ? Math.round(value) : Math.round(value * 100) / 100);
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      }
    }

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);

  return display;
}
