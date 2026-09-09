import { useEffect, useRef, useState } from "react";
import Map from "ol/Map";
import View from "ol/View";
import ImageLayer from "ol/layer/Image";
import ImageStatic from "ol/source/ImageStatic";
import Projection from "ol/proj/Projection";
import ScaleLine from "ol/control/ScaleLine";
import type MapBrowserEvent from "ol/MapBrowserEvent";
import "ol/ol.css";

import { sceneImageUrl, sceneMaskUrl } from "../api/client";
import { disablePixelSmoothing } from "../utils/olRendering";
import "./MapView.css";

interface MapViewProps {
  sceneId: string;
  width: number;
  height: number;
  gsdMetres: number;
  maskOpacity: number;
}

interface PointerReadout {
  xMetres: number;
  yMetres: number;
  row: number;
  col: number;
}

export default function MapView({ sceneId, width, height, gsdMetres, maskOpacity }: MapViewProps) {
  const mapDivRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const baseLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const maskLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const [pointer, setPointer] = useState<PointerReadout | null>(null);

  // Create the map once.
  useEffect(() => {
    if (!mapDivRef.current) return;

    const baseLayer = new ImageLayer<ImageStatic>();
    const maskLayer = new ImageLayer<ImageStatic>({ opacity: maskOpacity });
    disablePixelSmoothing(maskLayer); // a class mask is categorical -- never smoothed, see olRendering.ts
    baseLayerRef.current = baseLayer;
    maskLayerRef.current = maskLayer;

    const scaleLine = new ScaleLine({ units: "metric", bar: false });

    const map = new Map({
      target: mapDivRef.current,
      layers: [baseLayer, maskLayer],
      controls: [scaleLine],
      view: new View({ zoom: 1 }), // replaced immediately once a scene is known
    });
    mapRef.current = map;

    function handlePointerMove(evt: MapBrowserEvent) {
      const [x, y] = evt.coordinate;
      const view = mapRef.current?.getView();
      const extent = view?.getProjection().getExtent();
      if (!extent) return;
      const [, , , extentHeight] = extent;
      if (x < extent[0] || x > extent[2] || y < extent[1] || y > extent[3]) {
        setPointer(null);
        return;
      }
      setPointer({
        xMetres: x,
        yMetres: y,
        col: Math.floor(x / gsdMetresRef.current),
        row: Math.floor((extentHeight - y) / gsdMetresRef.current),
      });
    }
    map.on("pointermove", handlePointerMove);
    map.getViewport().addEventListener("pointerleave", () => setPointer(null));

    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // gsdMetres changes only when the scene changes -- keep a ref so the
  // pointermove closure (registered once) always reads the current value.
  const gsdMetresRef = useRef(gsdMetres);
  gsdMetresRef.current = gsdMetres;

  // Swap in the new scene's sources/extent/view whenever the scene changes.
  useEffect(() => {
    const map = mapRef.current;
    const baseLayer = baseLayerRef.current;
    const maskLayer = maskLayerRef.current;
    if (!map || !baseLayer || !maskLayer || !width || !height) return;

    const widthM = width * gsdMetres;
    const heightM = height * gsdMetres;
    const extent: [number, number, number, number] = [0, 0, widthM, heightM];

    const projection = new Projection({
      code: `scene-local:${sceneId}`,
      units: "m",
      extent,
    });

    baseLayer.setSource(
      new ImageStatic({ url: sceneImageUrl(sceneId), projection, imageExtent: extent }),
    );
    maskLayer.setSource(
      new ImageStatic({ url: sceneMaskUrl(sceneId), projection, imageExtent: extent }),
    );

    const view = new View({ projection, extent, showFullExtent: true });
    map.setView(view);

    // view.fit() does a "contain" fit -- it picks whichever axis is more
    // constraining and letterboxes the other, which is why a square scene
    // used to leave a wide dead band on a widescreen canvas. Instead, fill
    // the canvas WIDTH exactly (the scene may crop top/bottom if the canvas
    // is taller than the scene's aspect ratio, which is fine -- the scene
    // is always square and the canvas is always wider than tall here).
    const size = map.getSize();
    if (size && size[0] > 0) {
      view.setResolution(widthM / size[0]);
      view.setCenter([widthM / 2, heightM / 2]);
    } else {
      view.fit(extent, { padding: [48, 48, 48, 48] });
    }
  }, [sceneId, width, height, gsdMetres]);

  useEffect(() => {
    maskLayerRef.current?.setOpacity(maskOpacity);
  }, [maskOpacity]);

  return (
    <div className="map-view instrument-frame scanlines">
      <div ref={mapDivRef} className="map-view__canvas" />
      <div className="map-view__coord-readout tabular">
        {pointer ? (
          <>
            <span>X {pointer.xMetres.toFixed(0).padStart(5, " ")} m</span>
            <span>Y {pointer.yMetres.toFixed(0).padStart(5, " ")} m</span>
            <span className="map-view__coord-readout-dim">
              px [{pointer.row}, {pointer.col}]
            </span>
          </>
        ) : (
          <span className="map-view__coord-readout-dim">pointer off-scene</span>
        )}
      </div>
    </div>
  );
}
