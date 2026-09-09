import type ImageLayer from "ol/layer/Image";
import type ImageSource from "ol/source/Image";
import type RenderEvent from "ol/render/Event";

/**
 * Forces nearest-neighbor (blocky) scaling instead of the browser's
 * default smooth interpolation, for exactly one layer.
 *
 * A class mask is categorical data -- every pixel is one of
 * api/scenes.py's class_color() palette entries, nothing else. The
 * browser's default smooth scaling blends adjacent pixels' RGB values at
 * every class boundary when upscaling a small raster (these masks are
 * 120x120, stretched to fill a much larger map canvas), which can
 * synthesize a color that was never in the palette and corresponds to no
 * real class -- confirmed by comparing a raw mask PNG (zero off-palette
 * pixels) against the same mask as OpenLayers actually renders it (a
 * stray magenta fleck at a hard class boundary). Nearest-neighbor
 * guarantees every rendered pixel is exactly one of the palette's real
 * colors, never an invented blend -- literally "no pixel, no claim"
 * applied to the pixel doing the claiming.
 *
 * Only ever call this for a categorical layer (a class mask). A
 * true-color preview (real, continuous-valued reflectance) should keep
 * smooth scaling -- that's a real image, not a set of discrete labels.
 */
export function disablePixelSmoothing(layer: ImageLayer<ImageSource>): void {
  layer.on("prerender", (event: RenderEvent) => {
    const ctx = event.context as CanvasRenderingContext2D | undefined;
    if (ctx) ctx.imageSmoothingEnabled = false;
  });
  layer.on("postrender", (event: RenderEvent) => {
    const ctx = event.context as CanvasRenderingContext2D | undefined;
    if (ctx) ctx.imageSmoothingEnabled = true;
  });
}
