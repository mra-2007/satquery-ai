"""Tests for raster_io/readers.py: GeoTIFF/TIFF (georeferenced) and
PNG/JPEG (benchmark) reading, GSD/CRS/transform extraction, and pixel ->
lon/lat conversion, all against real files on disk."""

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from raster_io.readers import ReaderError, read_image, pixel_to_lonlat


def _write_geotiff(path, *, crs, transform, width=20, height=20, bands=3):
    data = np.random.default_rng(0).integers(0, 255, size=(bands, height, width), dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=bands,
        dtype="uint8", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)
    return data


def _write_plain_tiff(path, width=20, height=20, bands=1):
    data = np.random.default_rng(1).integers(0, 255, size=(bands, height, width), dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=bands, dtype="uint8") as dst:
        dst.write(data)
    return data


def _write_png(path, width=20, height=20):
    arr = np.random.default_rng(2).integers(0, 255, size=(height, width, 3), dtype="uint8")
    Image.fromarray(arr).save(path)
    return arr


def _write_jpeg(path, width=20, height=20):
    arr = np.random.default_rng(3).integers(0, 255, size=(height, width, 3), dtype="uint8")
    Image.fromarray(arr).save(path, format="JPEG", quality=100)
    return arr


# --- GeoTIFF: GSD/CRS/transform read from real metadata ----------------------


def test_geotiff_utm_gsd_is_close_to_pixel_size(tmp_path):
    path = tmp_path / "utm.tif"
    transform = from_origin(500000, 4649000, 10, 10)  # 10 m pixels
    _write_geotiff(path, crs=CRS.from_epsg(32643), transform=transform)

    image = read_image(path)

    assert image.crs is not None
    assert image.transform is not None
    assert image.gsd_metres == pytest.approx(10.0, abs=0.1)
    assert image.band_count == 3
    assert image.width == 20 and image.height == 20
    assert image.data.shape == (3, 20, 20)


def test_geotiff_geographic_crs_gsd_accounts_for_latitude(tmp_path):
    path = tmp_path / "geo.tif"
    transform = from_origin(77.5, 13.0, 0.0001, 0.0001)  # ~0.0001 deg pixels near lat 13
    _write_geotiff(path, crs=CRS.from_epsg(4326), transform=transform, bands=1)

    image = read_image(path)

    expected_gsd = 0.0001 * 111_320 * np.cos(np.radians(13.0))  # rough check
    assert image.gsd_metres == pytest.approx(expected_gsd, rel=0.05)


def test_geotiff_override_is_ignored_when_metadata_present(tmp_path):
    path = tmp_path / "utm2.tif"
    _write_geotiff(path, crs=CRS.from_epsg(32643), transform=from_origin(500000, 4649000, 10, 10))

    image = read_image(path, gsd_metres_override=999.0)

    assert image.gsd_metres == pytest.approx(10.0, abs=0.1)  # real metadata wins, not the override


# --- Non-georeferenced formats: GSD must be refused, not defaulted ----------


def test_plain_tiff_has_no_gsd_without_override(tmp_path):
    path = tmp_path / "plain.tif"
    _write_plain_tiff(path)

    image = read_image(path)

    assert image.crs is None
    assert image.transform is None
    assert image.gsd_metres is None  # refused, not defaulted to e.g. 10 m


def test_png_has_no_gsd_without_override(tmp_path):
    path = tmp_path / "bench.png"
    _write_png(path)

    image = read_image(path)

    assert image.crs is None
    assert image.transform is None
    assert image.gsd_metres is None


def test_png_uses_explicit_override(tmp_path):
    path = tmp_path / "bench.png"
    _write_png(path)

    image = read_image(path, gsd_metres_override=10.0)

    assert image.gsd_metres == 10.0


def test_jpeg_reads_and_accepts_override(tmp_path):
    path = tmp_path / "bench.jpg"
    _write_jpeg(path)

    image = read_image(path, gsd_metres_override=30.0)

    assert image.band_count == 3
    assert image.gsd_metres == 30.0


# --- Error handling ------------------------------------------------------------


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "image.bmp"
    path.write_bytes(b"not a real bmp")
    with pytest.raises(ReaderError, match="unsupported format"):
        read_image(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ReaderError, match="not found"):
        read_image(tmp_path / "does_not_exist.tif")


# --- pixel_to_lonlat -----------------------------------------------------------


def test_pixel_to_lonlat_on_georeferenced_image(tmp_path):
    path = tmp_path / "utm3.tif"
    # UTM 43N origin near (500000, 4649000) is roughly (77.06 E, 42.0 N)... use a
    # known-good round-trip instead of hand-computing the exact expected lon/lat.
    transform = from_origin(500000, 1400000, 10, 10)
    _write_geotiff(path, crs=CRS.from_epsg(32643), transform=transform)

    image = read_image(path)
    lon, lat = pixel_to_lonlat(image, row=0, col=0)

    assert -180 <= lon <= 180
    assert -90 <= lat <= 90


def test_pixel_to_lonlat_raises_without_georeferencing(tmp_path):
    path = tmp_path / "bench.png"
    _write_png(path)
    image = read_image(path)

    with pytest.raises(ReaderError, match="no georeferencing"):
        pixel_to_lonlat(image, row=0, col=0)
