import { useEffect, useRef, useState } from "react";
import Map from "ol/Map";
import View from "ol/View";
import ImageLayer from "ol/layer/Image";
import ImageStatic from "ol/source/ImageStatic";
import Projection from "ol/proj/Projection";
import ScaleLine from "ol/control/ScaleLine";
import { getRenderPixel } from "ol/render";
import type RenderEvent from "ol/render/Event";
import "ol/ol.css";

import { getSceneCrossModal, sceneCrossModalMaskUrl, sceneImageUrl } from "../api/client";
import type { CrossModalSensor, CrossModalSummary } from "../api/types";
import "./CrossModalView.css";

interface CrossModalViewProps {
  sceneId: string;
  width: number;
  height: number;
  gsdMetres: number;
}

const SENSORS: CrossModalSensor[] = ["optical", "sar", "fused"];

export default function CrossModalView({ sceneId, width, height, gsdMetres }: CrossModalViewProps) {
  const mapDivRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const baseLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const maskLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const dividerPercentRef = useRef(50);

  const [dividerPercent, setDividerPercent] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [sensor, setSensor] = useState<CrossModalSensor>("fused");
  const [cloudSimulation, setCloudSimulation] = useState(false);

  const [analysis, setAnalysis] = useState<CrossModalSummary | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Create the map once -- same "layer swipe" technique as ChangeView.tsx
  // (ol/render's prerender/postrender clip hooks): BASE (the true-color
  // preview) is the always-full left layer, the toggled sensor's predicted
  // MASK is clipped to whatever lies right of the divider.
  useEffect(() => {
    if (!mapDivRef.current) return;

    const baseLayer = new ImageLayer<ImageStatic>();
    const maskLayer = new ImageLayer<ImageStatic>();
    baseLayerRef.current = baseLayer;
    maskLayerRef.current = maskLayer;

    function clipToRightOfDivider(event: RenderEvent) {
      const ctx = event.context as CanvasRenderingContext2D;
      const mapSize = mapRef.current?.getSize();
      if (!mapSize) return;
      const dividerPx = mapSize[0] * (dividerPercentRef.current / 100);
      const topLeft = getRenderPixel(event, [dividerPx, 0]);
      const topRight = getRenderPixel(event, [mapSize[0], 0]);
      const bottomLeft = getRenderPixel(event, [dividerPx, mapSize[1]]);
      const bottomRight = getRenderPixel(event, [mapSize[0], mapSize[1]]);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(topLeft[0], topLeft[1]);
      ctx.lineTo(bottomLeft[0], bottomLeft[1]);
      ctx.lineTo(bottomRight[0], bottomRight[1]);
      ctx.lineTo(topRight[0], topRight[1]);
      ctx.closePath();
      ctx.clip();
    }
    function restore(event: RenderEvent) {
      (event.context as CanvasRenderingContext2D).restore();
    }
    maskLayer.on("prerender", clipToRightOfDivider);
    maskLayer.on("postrender", restore);

    const scaleLine = new ScaleLine({ units: "metric", bar: false });

    const map = new Map({
      target: mapDivRef.current,
      layers: [baseLayer, maskLayer],
      controls: [scaleLine],
      view: new View({ zoom: 1 }), // replaced immediately once a scene is known
    });
    mapRef.current = map;

    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
    };
  }, []);

  // Swap in the new scene/sensor/cloud-simulation sources whenever any of
  // them change. The base (left) layer always reflects cloud_simulation
  // directly -- the clouded top rows are visible there regardless of which
  // sensor is toggled -- while the mask (right) layer is whichever sensor
  // path is currently selected, segmented over that same (possibly
  // clouded) stack.
  useEffect(() => {
    const map = mapRef.current;
    const baseLayer = baseLayerRef.current;
    const maskLayer = maskLayerRef.current;
    if (!map || !baseLayer || !maskLayer || !width || !height) return;

    const widthM = width * gsdMetres;
    const heightM = height * gsdMetres;
    const extent: [number, number, number, number] = [0, 0, widthM, heightM];

    const projection = new Projection({ code: `scene-local-cross-modal:${sceneId}`, units: "m", extent });

    baseLayer.setSource(
      new ImageStatic({
        url: sceneImageUrl(sceneId, undefined, cloudSimulation),
        projection,
        imageExtent: extent,
      }),
    );
    maskLayer.setSource(
      new ImageStatic({
        url: sceneCrossModalMaskUrl(sceneId, sensor, cloudSimulation),
        projection,
        imageExtent: extent,
      }),
    );

    const view = new View({ projection, extent, showFullExtent: true });
    map.setView(view);

    // Fill the canvas width exactly, same fix as MapView/ChangeView.
    const size = map.getSize();
    if (size && size[0] > 0) {
      view.setResolution(widthM / size[0]);
      view.setCenter([widthM / 2, heightM / 2]);
    } else {
      view.fit(extent, { padding: [48, 48, 48, 48] });
    }
  }, [sceneId, width, height, gsdMetres, sensor, cloudSimulation]);

  // The findings table and per-sensor USABLE/INSUFFICIENT verdicts --
  // independent of the map layers, refetched whenever the scene or the
  // cloud-simulation toggle changes (the sensor toggle alone doesn't
  // change tools/cross_modal.py's own per-class comparison, only which
  // mask layer is drawn).
  useEffect(() => {
    let cancelled = false;
    setAnalysisLoading(true);
    setAnalysisError(null);
    getSceneCrossModal(sceneId, cloudSimulation)
      .then((result) => {
        if (!cancelled) setAnalysis(result);
      })
      .catch((err) => {
        if (!cancelled) setAnalysisError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setAnalysisLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sceneId, cloudSimulation]);

  function moveDividerTo(clientX: number) {
    const rect = mapDivRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    dividerPercentRef.current = pct;
    setDividerPercent(pct);
    maskLayerRef.current?.changed();
  }

  function handleDividerPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  }
  function handleDividerPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    moveDividerTo(e.clientX);
  }
  function handleDividerPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.releasePointerCapture(e.pointerId);
    setDragging(false);
  }

  return (
    <div className="cross-modal-view instrument-frame scanlines">
      <div ref={mapDivRef} className="cross-modal-view__canvas" />

      <div className="cross-modal-view__side-badge cross-modal-view__side-badge--base">
        <span className="label">BASE</span>
      </div>
      <div className="cross-modal-view__side-badge cross-modal-view__side-badge--mask">
        <span className="label">{sensor.toUpperCase()} MASK</span>
      </div>

      <div
        className={"cross-modal-view__divider" + (dragging ? " cross-modal-view__divider--dragging" : "")}
        style={{ left: `${dividerPercent}%` }}
        onPointerDown={handleDividerPointerDown}
        onPointerMove={handleDividerPointerMove}
        onPointerUp={handleDividerPointerUp}
      >
        <span className="cross-modal-view__divider-line" />
        <span className="cross-modal-view__divider-handle">⇔</span>
      </div>

      <div className="cross-modal-view__controls">
        <div className="cross-modal-view__sensor-toggle">
          {SENSORS.map((option) => (
            <button
              key={option}
              type="button"
              className={sensor === option ? "is-active" : ""}
              onClick={() => setSensor(option)}
            >
              {option.toUpperCase()}
              {analysis && (
                <span
                  className={
                    "cross-modal-view__sensor-dot" +
                    (analysis.sensor_status[option] === "INSUFFICIENT"
                      ? " cross-modal-view__sensor-dot--insufficient"
                      : " cross-modal-view__sensor-dot--usable")
                  }
                />
              )}
            </button>
          ))}
        </div>

        <label className="cross-modal-view__cloud-toggle">
          <input
            type="checkbox"
            checked={cloudSimulation}
            onChange={(e) => setCloudSimulation(e.target.checked)}
          />
          <span className="label">SIMULATE CLOUD COVER</span>
        </label>
      </div>

      <CrossModalFindingsPanel analysis={analysis} loading={analysisLoading} error={analysisError} />
    </div>
  );
}

function CrossModalFindingsPanel({
  analysis,
  loading,
  error,
}: {
  analysis: CrossModalSummary | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading && !analysis) {
    return (
      <div className="cross-modal-findings">
        <span className="label">CROSS-MODAL FINDINGS</span>
        <p className="cross-modal-findings__status">Comparing sensor paths...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="cross-modal-findings">
        <span className="label">CROSS-MODAL FINDINGS</span>
        <p className="cross-modal-findings__status cross-modal-findings__status--error">{error}</p>
      </div>
    );
  }

  if (!analysis) return null;

  return (
    <div className="cross-modal-findings">
      <div className="cross-modal-findings__header">
        <span className="label">CROSS-MODAL FINDINGS</span>
        <span className="tabular cross-modal-findings__count">{analysis.findings.length} class(es)</span>
      </div>

      <div className="cross-modal-findings__status-row">
        {SENSORS.map((option) => (
          <span
            key={option}
            className={
              "cross-modal-findings__status-chip" +
              (analysis.sensor_status[option] === "INSUFFICIENT"
                ? " cross-modal-findings__status-chip--insufficient"
                : " cross-modal-findings__status-chip--usable")
            }
          >
            {option.toUpperCase()} {analysis.sensor_status[option]}
          </span>
        ))}
      </div>

      {analysis.findings.length === 0 ? (
        <p className="cross-modal-findings__status">No land-cover classes were detected by any sensor mode.</p>
      ) : (
        <div className="cross-modal-findings__rows">
          <div className="cross-modal-findings__row cross-modal-findings__row--head">
            <span className="label">CLASS</span>
            <span className="label cross-modal-findings__num-head">OPT</span>
            <span className="label cross-modal-findings__num-head">SAR</span>
            <span className="label cross-modal-findings__num-head">FUS</span>
          </div>
          {analysis.findings.map((finding) => (
            <div className="cross-modal-findings__row" key={finding.class_id} title={finding.attribution}>
              <span className="cross-modal-findings__name">{finding.class_name}</span>
              <span
                className={
                  "tabular cross-modal-findings__num" +
                  (finding.detected_by.includes("optical") ? " cross-modal-findings__num--hit" : "")
                }
              >
                {finding.optical_area_ha > 0 ? finding.optical_area_ha.toFixed(1) : "–"}
              </span>
              <span
                className={
                  "tabular cross-modal-findings__num" +
                  (finding.detected_by.includes("sar") ? " cross-modal-findings__num--hit" : "")
                }
              >
                {finding.sar_area_ha > 0 ? finding.sar_area_ha.toFixed(1) : "–"}
              </span>
              <span
                className={
                  "tabular cross-modal-findings__num" +
                  (finding.detected_by.includes("fused") ? " cross-modal-findings__num--hit" : "")
                }
              >
                {finding.fused_area_ha > 0 ? finding.fused_area_ha.toFixed(1) : "–"}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="cross-modal-findings__summary">{analysis.summary}</p>
    </div>
  );
}
