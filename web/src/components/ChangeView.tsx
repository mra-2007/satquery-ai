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

import { sceneChangeMaskUrl, sceneImageUrl } from "../api/client";
import "./ChangeView.css";

interface ChangeViewProps {
  sceneId: string;
  width: number;
  height: number;
  gsdMetres: number;
  beforeDate: string | null;
  afterDate: string | null;
}

export default function ChangeView({ sceneId, width, height, gsdMetres, beforeDate, afterDate }: ChangeViewProps) {
  const mapDivRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const beforeLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const afterLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const changeMaskLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const dividerPercentRef = useRef(50);

  const [dividerPercent, setDividerPercent] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [showChangeMask, setShowChangeMask] = useState(false);

  // Create the map once: BEFORE is the always-full base layer; AFTER is
  // clipped to whatever lies right of the divider (ol/render's
  // prerender/postrender hooks, the standard OL "layer swipe" technique --
  // see https://openlayers.org/en/latest/examples/layer-swipe.html) so the
  // divider reveals AFTER on the right and BEFORE shows through on the
  // left. Both layers share this one View, so pan/zoom are inherently
  // synchronised -- there is no separate sync step to get wrong.
  useEffect(() => {
    if (!mapDivRef.current) return;

    const beforeLayer = new ImageLayer<ImageStatic>();
    const afterLayer = new ImageLayer<ImageStatic>();
    const changeMaskLayer = new ImageLayer<ImageStatic>({ visible: false, opacity: 0.9 });
    beforeLayerRef.current = beforeLayer;
    afterLayerRef.current = afterLayer;
    changeMaskLayerRef.current = changeMaskLayer;

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
    afterLayer.on("prerender", clipToRightOfDivider);
    afterLayer.on("postrender", restore);

    const scaleLine = new ScaleLine({ units: "metric", bar: false });

    const map = new Map({
      target: mapDivRef.current,
      layers: [beforeLayer, afterLayer, changeMaskLayer],
      controls: [scaleLine],
      view: new View({ zoom: 1 }), // replaced immediately once a scene is known
    });
    mapRef.current = map;

    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
    };
  }, []);

  // Swap in the new scene's sources/extent/view whenever the scene changes.
  useEffect(() => {
    const map = mapRef.current;
    const beforeLayer = beforeLayerRef.current;
    const afterLayer = afterLayerRef.current;
    const changeMaskLayer = changeMaskLayerRef.current;
    if (!map || !beforeLayer || !afterLayer || !changeMaskLayer || !width || !height) return;

    const widthM = width * gsdMetres;
    const heightM = height * gsdMetres;
    const extent: [number, number, number, number] = [0, 0, widthM, heightM];

    const projection = new Projection({ code: `scene-local-change:${sceneId}`, units: "m", extent });

    beforeLayer.setSource(
      new ImageStatic({ url: sceneImageUrl(sceneId, "before"), projection, imageExtent: extent }),
    );
    afterLayer.setSource(
      new ImageStatic({ url: sceneImageUrl(sceneId, "after"), projection, imageExtent: extent }),
    );
    changeMaskLayer.setSource(
      new ImageStatic({ url: sceneChangeMaskUrl(sceneId), projection, imageExtent: extent }),
    );

    const view = new View({ projection, extent, showFullExtent: true });
    map.setView(view);

    // Fill the canvas width exactly, same fix as MapView's ANALYZE map --
    // view.fit() would letterbox a square-ish scene on a wide canvas.
    const size = map.getSize();
    if (size && size[0] > 0) {
      view.setResolution(widthM / size[0]);
      view.setCenter([widthM / 2, heightM / 2]);
    } else {
      view.fit(extent, { padding: [48, 48, 48, 48] });
    }
  }, [sceneId, width, height, gsdMetres]);

  useEffect(() => {
    changeMaskLayerRef.current?.setVisible(showChangeMask);
  }, [showChangeMask]);

  function moveDividerTo(clientX: number) {
    const rect = mapDivRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    dividerPercentRef.current = pct;
    setDividerPercent(pct);
    afterLayerRef.current?.changed();
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
    <div className="change-view instrument-frame scanlines">
      <div ref={mapDivRef} className="change-view__canvas" />

      <div className="change-view__date-badge change-view__date-badge--before">
        <span className="label">BEFORE</span>
        {beforeDate && <span className="tabular change-view__date-value">{beforeDate}</span>}
      </div>
      <div className="change-view__date-badge change-view__date-badge--after">
        <span className="label">AFTER</span>
        {afterDate && <span className="tabular change-view__date-value">{afterDate}</span>}
      </div>

      <div
        className={"change-view__divider" + (dragging ? " change-view__divider--dragging" : "")}
        style={{ left: `${dividerPercent}%` }}
        onPointerDown={handleDividerPointerDown}
        onPointerMove={handleDividerPointerMove}
        onPointerUp={handleDividerPointerUp}
      >
        <span className="change-view__divider-line" />
        <span className="change-view__divider-handle">⇔</span>
      </div>

      <label className="change-view__mask-toggle">
        <input
          type="checkbox"
          checked={showChangeMask}
          onChange={(e) => setShowChangeMask(e.target.checked)}
        />
        <span className="label">CHANGE MASK</span>
      </label>
    </div>
  );
}
