"""Deterministic grounding: resolve a natural-language referring
expression like "the water body in the north-west" to ONE connected
region of the class-ID mask, and return that region's polygon and
bounding box in both pixel and geographic coordinates.

Per CLAUDE.md's "No pixel, no claim": the noun is resolved to a class via
agent/vocabulary.py's synonym map (the same map agent/planner.py's
keyword fallback and every eval script use -- one vocabulary, everywhere),
and the region itself is found and measured with the same tools every
other geometry operation in this codebase uses: scipy.ndimage.label
(evidence/ops.py's own connected-component convention) for the region,
and evidence/ops.py's own pixel-area arithmetic for its size. No language
model is involved anywhere in this module -- there is nothing here for an
LLM to get right or wrong, so unlike tools/caption.py there is no offline
fallback to design around.

Supported qualifiers (case-insensitive, anywhere in the phrase): the
compass directions north/south/east/west and their intercardinal
compounds (northwest/north-west/north west, etc.), central/center/centre,
and the superlatives largest/biggest and smallest/tiniest. When more than
one region of the resolved class exists:
  - a superlative, if present, wins outright (global largest/smallest by
    area) -- a direction word elsewhere in the same phrase is ignored.
  - otherwise a direction/centrality word selects the region whose
    centroid is closest to that side of the image (or its centre).
  - with neither qualifier, the largest region is returned (the most
    salient default, not an error) -- see GroundingResult.qualifier to
    see whether a query's qualifier was actually understood.

Geographic coordinates require `metadata` to carry a real affine
`transform` and `crs` (as raster_io.readers.RasterImage does) -- pass
them as metadata["transform"]/metadata["crs"] when available. Without
them, every geographic field (`geo_bbox`, `geo_polygon`,
`centroid_lonlat`) is None, per the same "refuse, don't default"
convention raster_io/readers.py uses for GSD -- this module never
invents a coordinate reference system.
"""

import re
from dataclasses import dataclass

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from shapely.geometry import shape as shapely_shape

from agent.vocabulary import SEGMENTATION_CLASSES, resolve_noun
from evidence.schema import Polygon

_CLASS_NAME_TO_ID = {name: i for i, name in enumerate(SEGMENTATION_CLASSES)}


class GroundingError(ValueError):
    """Raised when the query's noun can't be resolved to a class, or the
    resolved class has no region anywhere in the mask."""


@dataclass
class GroundingResult:
    class_id: int
    class_name: str
    area_ha: float
    candidate_count: int  # how many regions of this class existed to choose from
    qualifier: dict[str, str | None]  # {"direction": ..., "superlative": ...} as parsed from the query
    centroid_pixel: tuple[float, float]  # (row, col)
    centroid_lonlat: tuple[float, float] | None
    pixel_bbox: tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max), inclusive
    geo_bbox: tuple[float, float, float, float] | None  # (min_lon, min_lat, max_lon, max_lat)
    pixel_polygon: Polygon
    geo_polygon: Polygon | None


# --- qualifier parsing --------------------------------------------------------

_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "north-west": (-1, -1), "north west": (-1, -1), "northwest": (-1, -1),
    "north-east": (-1, 1), "north east": (-1, 1), "northeast": (-1, 1),
    "south-west": (1, -1), "south west": (1, -1), "southwest": (1, -1),
    "south-east": (1, 1), "south east": (1, 1), "southeast": (1, 1),
    "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1),
    "central": (0, 0), "center": (0, 0), "centre": (0, 0),
}
_SUPERLATIVES: dict[str, str] = {
    "largest": "largest", "biggest": "largest",
    "smallest": "smallest", "tiniest": "smallest",
}


def _parse_qualifiers(query: str) -> tuple[str, tuple[int, int] | None, str | None]:
    """Strips the first direction phrase and first superlative found from
    `query` (longest direction phrases matched first, so "north-west"
    isn't mistaken for a bare "north"), returning (remaining_text,
    direction_vector, superlative). Either or both may be None."""
    remaining = query.lower()

    direction_vector = None
    for phrase in sorted(_DIRECTION_VECTORS, key=len, reverse=True):
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, remaining):
            direction_vector = _DIRECTION_VECTORS[phrase]
            remaining = re.sub(pattern, " ", remaining, count=1)
            break

    superlative = None
    for word, canonical in _SUPERLATIVES.items():
        pattern = rf"\b{word}\b"
        if re.search(pattern, remaining):
            superlative = canonical
            remaining = re.sub(pattern, " ", remaining, count=1)
            break

    return remaining, direction_vector, superlative


# --- region enumeration and selection ----------------------------------------


def _pixel_area_ha(mask: np.ndarray, metadata: dict) -> float:
    gsd = metadata["gsd_metres"]
    return (gsd * gsd) / 10_000.0


def _regions_for_class(mask: np.ndarray, class_id: int) -> list[np.ndarray]:
    labeled, n_components = ndimage.label(mask == class_id)
    return [labeled == label_id for label_id in range(1, n_components + 1)]


def _select_region(
    regions: list[np.ndarray],
    areas_ha: list[float],
    centroids: list[tuple[float, float]],
    direction_vector: tuple[int, int] | None,
    superlative: str | None,
    mask_shape: tuple[int, int],
) -> int:
    """Returns the index into `regions` to use -- see the module
    docstring's qualifier-priority rules."""
    if superlative == "largest":
        return max(range(len(regions)), key=lambda i: areas_ha[i])
    if superlative == "smallest":
        return min(range(len(regions)), key=lambda i: areas_ha[i])

    if direction_vector is not None:
        height, width = mask_shape
        row_sign, col_sign = direction_vector
        target_row = 0 if row_sign < 0 else (height - 1) if row_sign > 0 else (height - 1) / 2
        target_col = 0 if col_sign < 0 else (width - 1) if col_sign > 0 else (width - 1) / 2

        def distance_to_target(i: int) -> float:
            row, col = centroids[i]
            return (row - target_row) ** 2 + (col - target_col) ** 2

        return min(range(len(regions)), key=distance_to_target)

    return max(range(len(regions)), key=lambda i: areas_ha[i])  # no qualifier: most salient default


# --- polygon/bbox extraction ---------------------------------------------------


def _pixel_bbox(region_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(region_mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _vectorize(region_mask: np.ndarray, transform: Affine) -> dict:
    """region_mask's polygon, with vertex coordinates in whatever space
    `transform` maps pixel (col, row) into. Picks the largest-area shape
    if rasterio's vectorizer returns more than one for this single
    connected component (rare -- only for shapes 4-connected but pinched
    at a single corner, which trace as separate simple polygons)."""
    shapes = [
        geom for geom, value in features.shapes(
            region_mask.astype(np.uint8), mask=region_mask, transform=transform
        )
        if value == 1
    ]
    if not shapes:
        raise GroundingError("selected region produced no polygon (this should not happen)")
    return max(shapes, key=lambda geom: shapely_shape(geom).area)


def _reproject_polygon(geojson_geometry: dict, crs: str) -> dict:
    """geojson_geometry's coordinates are in `crs` units -- returns the
    same Polygon reprojected to EPSG:4326 (lon/lat), using the same
    rasterio.warp.transform raster_io.readers.pixel_to_lonlat uses."""

    def reproject_ring(ring: list) -> list[list[float]]:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
        return [[lon, lat] for lon, lat in zip(lons, lats)]

    return {
        "type": geojson_geometry["type"],
        "coordinates": [reproject_ring(ring) for ring in geojson_geometry["coordinates"]],
    }


def _bbox_from_polygon(geometry: dict) -> tuple[float, float, float, float]:
    xs = [point[0] for ring in geometry["coordinates"] for point in ring]
    ys = [point[1] for ring in geometry["coordinates"] for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


# --- main entry point ---------------------------------------------------------


def ground(mask: np.ndarray, metadata: dict, query: str) -> GroundingResult:
    """Resolve `query` (e.g. "the water body in the north-west") to one
    connected region of `mask` and return its geometry.

    Raises GroundingError if the query's noun doesn't resolve to any of
    agent.vocabulary.SEGMENTATION_CLASSES's 19 classes, or the resolved
    class has no region anywhere in `mask`.
    """
    remaining_text, direction_vector, superlative = _parse_qualifiers(query)

    class_name = resolve_noun(remaining_text)
    if class_name is None:
        raise GroundingError(f"could not resolve a class from {query!r}")
    class_id = _CLASS_NAME_TO_ID[class_name]

    regions = _regions_for_class(mask, class_id)
    if not regions:
        raise GroundingError(f"no {class_name!r} region found in this image")

    pixel_area_ha = _pixel_area_ha(mask, metadata)
    areas_ha = [float(region.sum()) * pixel_area_ha for region in regions]
    centroids = [ndimage.center_of_mass(region) for region in regions]

    selected = _select_region(regions, areas_ha, centroids, direction_vector, superlative, mask.shape)
    region_mask = regions[selected]

    pixel_geometry = _vectorize(region_mask, Affine.identity())
    pixel_polygon = Polygon.model_validate(pixel_geometry)
    pixel_bbox = _pixel_bbox(region_mask)

    transform = metadata.get("transform")
    crs = metadata.get("crs")
    geo_polygon = None
    geo_bbox = None
    centroid_lonlat = None
    if transform is not None and crs is not None:
        native_geometry = _vectorize(region_mask, transform)
        geo_geometry = _reproject_polygon(native_geometry, crs)
        geo_polygon = Polygon.model_validate(geo_geometry)
        geo_bbox = _bbox_from_polygon(geo_geometry)
        row, col = centroids[selected]
        x, y = transform @ (col + 0.5, row + 0.5)
        lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
        centroid_lonlat = (lons[0], lats[0])

    return GroundingResult(
        class_id=class_id,
        class_name=class_name,
        area_ha=areas_ha[selected],
        candidate_count=len(regions),
        qualifier={
            "direction": next((k for k, v in _DIRECTION_VECTORS.items() if v == direction_vector), None)
            if direction_vector is not None else None,
            "superlative": superlative,
        },
        centroid_pixel=centroids[selected],
        centroid_lonlat=centroid_lonlat,
        pixel_bbox=pixel_bbox,
        geo_bbox=geo_bbox,
        pixel_polygon=pixel_polygon,
        geo_polygon=geo_polygon,
    )
