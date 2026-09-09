import { useState } from "react";

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
        className="upload-control__trigger"
        type="button"
        onClick={() => setOpen((value) => !value)}
      >
        + UPLOAD
      </button>

      {open && (
        <div className="upload-control__popover">
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
        </div>
      )}
    </div>
  );
}
