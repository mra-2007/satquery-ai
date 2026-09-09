import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { uploadScene } from "../api/client";
import type { UploadResponse } from "../api/types";
import "./UploadControl.css";

type Mode = "single" | "pair";
type PairKind = "cross_modal" | "change";

interface UploadControlProps {
  onUploaded: (response: UploadResponse) => void;
}

export default function UploadControl({ onUploaded }: UploadControlProps) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [anchor, setAnchor] = useState({ top: 0, left: 0 });

  const [mode, setMode] = useState<Mode>("single");
  const [pairKind, setPairKind] = useState<PairKind>("cross_modal");

  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState("optical");
  const [gsd, setGsd] = useState("10");

  const [file2, setFile2] = useState<File | null>(null);
  const [modality2, setModality2] = useState("sar");
  const [gsd2, setGsd2] = useState("10");

  const [beforeDate, setBeforeDate] = useState("");
  const [afterDate, setAfterDate] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    setFile(null);
    setFile2(null);
    setBeforeDate("");
    setAfterDate("");
    setError(null);
  }

  function openPanel() {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) setAnchor({ top: rect.bottom + 8, left: rect.left });
    setOpen(true);
  }

  // Portaled to document.body (see the JSX below), so this panel is no
  // longer a descendant of TopBar's own stacking context -- that was the
  // actual bug: a z-index set on a descendant is capped by whatever
  // stacking order its positioned ancestor sits in, so no z-index inside
  // TopBar could ever out-rank CommandBar's, which sits outside it
  // entirely. Being a true sibling of everything else means a plain
  // z-index comparison (see --z-modal in theme.css) is now sufficient, and
  // it needs its own outside-click/Escape handling since it's no longer a
  // simple CSS-anchored popover.
  useEffect(() => {
    if (!open) return;

    function handlePointerDown(e: MouseEvent) {
      const target = e.target as Node;
      if (panelRef.current?.contains(target)) return;
      if (triggerRef.current?.contains(target)) return;
      setOpen(false);
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    function handleResize() {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (rect) setAnchor({ top: rect.bottom + 8, left: rect.left });
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", handleResize);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleResize);
    };
  }, [open]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || (mode === "pair" && !file2)) return;

    setSubmitting(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("modality", modality);
    if (gsd.trim()) formData.append("gsd_metres", gsd);
    if (mode === "pair" && file2) {
      formData.append("file2", file2);
      formData.append("modality2", modality2);
      if (gsd2.trim()) formData.append("gsd_metres2", gsd2);
      formData.append("pair_kind", pairKind);
      if (pairKind === "change") {
        if (beforeDate.trim()) formData.append("before_date", beforeDate.trim());
        if (afterDate.trim()) formData.append("after_date", afterDate.trim());
      }
    }

    try {
      const response = await uploadScene(formData);
      if (!response.ok) {
        setError(response.reason ?? "Upload was rejected.");
        return;
      }
      onUploaded(response);
      setOpen(false);
      reset();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="upload-control">
      <button
        ref={triggerRef}
        className="upload-control__trigger"
        type="button"
        onClick={() => (open ? setOpen(false) : openPanel())}
      >
        + UPLOAD
      </button>

      {open &&
        createPortal(
          <div ref={panelRef} className="upload-control__panel" style={{ top: anchor.top, left: anchor.left }}>
            <div className="upload-control__panel-header">
              <span className="label">UPLOAD SCENE</span>
              <button
                type="button"
                className="upload-control__close"
                onClick={() => setOpen(false)}
                aria-label="Close upload panel"
              >
                ×
              </button>
            </div>

            <div className="upload-control__mode-toggle">
              <button
                type="button"
                className={mode === "single" ? "is-active" : ""}
                onClick={() => setMode("single")}
              >
                SINGLE
              </button>
              <button
                type="button"
                className={mode === "pair" ? "is-active" : ""}
                onClick={() => setMode("pair")}
              >
                PAIR
              </button>
            </div>

            <form className="upload-control__form" onSubmit={handleSubmit}>
              {mode === "pair" && (
                <label className="upload-control__field">
                  <span className="label">PAIR TYPE</span>
                  <select value={pairKind} onChange={(e) => setPairKind(e.target.value as PairKind)}>
                    <option value="cross_modal">Cross-modal (optical + SAR)</option>
                    <option value="change">Change (bi-temporal)</option>
                  </select>
                </label>
              )}

              <fieldset className="upload-control__image-group">
                <legend className="label">{mode === "pair" ? "IMAGE A" : "IMAGE"}</legend>
                <input
                  type="file"
                  accept="image/png,image/jpeg,.tif,.tiff"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <div className="upload-control__row">
                  <select value={modality} onChange={(e) => setModality(e.target.value)}>
                    <option value="optical">optical</option>
                    <option value="sar">sar</option>
                  </select>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="GSD (m)"
                    value={gsd}
                    onChange={(e) => setGsd(e.target.value)}
                  />
                </div>
                {mode === "pair" && pairKind === "change" && (
                  <div className="upload-control__row">
                    <input
                      type="text"
                      placeholder="Date (optional, e.g. 2019-03)"
                      value={beforeDate}
                      onChange={(e) => setBeforeDate(e.target.value)}
                    />
                  </div>
                )}
              </fieldset>

              {mode === "pair" && (
                <fieldset className="upload-control__image-group">
                  <legend className="label">IMAGE B</legend>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,.tif,.tiff"
                    onChange={(e) => setFile2(e.target.files?.[0] ?? null)}
                  />
                  <div className="upload-control__row">
                    <select value={modality2} onChange={(e) => setModality2(e.target.value)}>
                      <option value="sar">sar</option>
                      <option value="optical">optical</option>
                    </select>
                    <input
                      type="number"
                      step="0.1"
                      placeholder="GSD (m)"
                      value={gsd2}
                      onChange={(e) => setGsd2(e.target.value)}
                    />
                  </div>
                  {pairKind === "change" && (
                    <div className="upload-control__row">
                      <input
                        type="text"
                        placeholder="Date (optional, e.g. 2023-11)"
                        value={afterDate}
                        onChange={(e) => setAfterDate(e.target.value)}
                      />
                    </div>
                  )}
                </fieldset>
              )}

              {error && <p className="upload-control__error">{error}</p>}

              <button
                type="submit"
                className="upload-control__submit"
                disabled={submitting || !file || (mode === "pair" && !file2)}
              >
                {submitting ? "UPLOADING…" : "UPLOAD"}
              </button>
            </form>
          </div>,
          document.body,
        )}
    </div>
  );
}
