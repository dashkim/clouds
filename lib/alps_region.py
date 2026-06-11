"""Shared Bavarian Alps crop, mask, and NetCDF loading helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np

from lib.netcdf import fill_value, mask_invalid

# Wider fetch box (irregular mask / 3D); display follows elevation mask inside it
DEFAULT_LON_MIN = 9.5
DEFAULT_LON_MAX = 14.2
DEFAULT_LAT_MIN = 47.5
DEFAULT_LAT_MAX = 48.25
DEFAULT_ELEV_MIN = 900.0

# Full Germany domain (local 2d_lonlat file coverage)
COUNTRY_LON_MIN = 4.5
COUNTRY_LON_MAX = 14.5
COUNTRY_LAT_MIN = 47.5
COUNTRY_LAT_MAX = 54.5

# Rectangular crop (Phase 1 exploration box)
RECT_LON_MIN = 10.0
RECT_LON_MAX = 13.0
RECT_LAT_MIN = 47.5
RECT_LAT_MAX = 48.2

# Small square patch (cloud-focused subregion near Zugspitze / Wetterstein)
PATCH_CENTER_LON = 11.0
PATCH_CENTER_LAT = 47.6
PATCH_HALF_SIZE = 0.11  # degrees; lat clipped at domain south edge (47.5°N)

# Eastern Alps / Bavarian Forest — high cloud variability in the 20-min window
# Domain lon max is ~14.496°E (no data to 15°E)
EAST_PATCH_LON_MIN = 13.0
EAST_PATCH_LON_MAX = 14.496
EAST_PATCH_LAT_MIN = 48.0
EAST_PATCH_LAT_MAX = 50.0
EAST_PATCH_CENTER_LON = (EAST_PATCH_LON_MIN + EAST_PATCH_LON_MAX) / 2.0
EAST_PATCH_CENTER_LAT = (EAST_PATCH_LAT_MIN + EAST_PATCH_LAT_MAX) / 2.0

# Mountainous core within the east patch (Bavarian/Bohemian Forest ridges)
EAST_CORE_LON_MIN = 13.38
EAST_CORE_LON_MAX = 13.68
EAST_CORE_LAT_MIN = 48.8125
EAST_CORE_LAT_MAX = 49.1125
EAST_CORE_CENTER_LON = (EAST_CORE_LON_MIN + EAST_CORE_LON_MAX) / 2.0
EAST_CORE_CENTER_LAT = (EAST_CORE_LAT_MIN + EAST_CORE_LAT_MAX) / 2.0

# Vertical exaggeration defaults (east box is ~10× wider than west patch)
PATCH_VERT_EXAG_WEST = 5.0
PATCH_VERT_EXAG_EAST = 10.0

PATCH_REGIONS = ("west", "east", "east_core")


def default_patch_vert_exag(region: str) -> float:
    if region in ("east", "east_core"):
        return PATCH_VERT_EXAG_EAST
    return PATCH_VERT_EXAG_WEST


def default_patch_stride(region: str) -> int:
    return 1 if region == "east_core" else (2 if region == "east" else 1)


def resolve_patch_bounds(
    *,
    region: str | None = None,
    center_lon: float | None = None,
    center_lat: float | None = None,
    half_size: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return lon/lat bounds and metric center (lon_c, lat_c)."""
    if region == "west":
        lomin, lomax, lamin, lamax = square_patch_bounds()
        return lomin, lomax, lamin, lamax, PATCH_CENTER_LON, PATCH_CENTER_LAT
    if region == "east":
        return (
            EAST_PATCH_LON_MIN, EAST_PATCH_LON_MAX,
            EAST_PATCH_LAT_MIN, EAST_PATCH_LAT_MAX,
            EAST_PATCH_CENTER_LON, EAST_PATCH_CENTER_LAT,
        )
    if region == "east_core":
        return (
            EAST_CORE_LON_MIN, EAST_CORE_LON_MAX,
            EAST_CORE_LAT_MIN, EAST_CORE_LAT_MAX,
            EAST_CORE_CENTER_LON, EAST_CORE_CENTER_LAT,
        )
    if region is not None:
        raise ValueError(f"Unknown region {region!r}; choose from {list(PATCH_REGIONS)}")
    if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
        clon = (lon_min + lon_max) / 2.0
        clat = (lat_min + lat_max) / 2.0
        return lon_min, lon_max, lat_min, lat_max, clon, clat
    clon = center_lon if center_lon is not None else PATCH_CENTER_LON
    clat = center_lat if center_lat is not None else PATCH_CENTER_LAT
    hs = half_size if half_size is not None else PATCH_HALF_SIZE
    lomin, lomax, lamin, lamax = square_patch_bounds(clon, clat, hs)
    return lomin, lomax, lamin, lamax, clon, clat


def square_patch_bounds(
    center_lon: float = PATCH_CENTER_LON,
    center_lat: float = PATCH_CENTER_LAT,
    half_size: float = PATCH_HALF_SIZE,
    *,
    lat_floor: float = 47.5,
) -> tuple[float, float, float, float]:
    """Return lon/lat bounds for a square patch, clamping latitude to the domain."""
    lat_min = max(lat_floor, center_lat - half_size)
    lat_max = center_lat + half_size
    lat_span = lat_max - lat_min
    lon_min = center_lon - lat_span / 2.0
    lon_max = center_lon + lat_span / 2.0
    return lon_min, lon_max, lat_min, lat_max


def crop_indices(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[slice, slice]:
    lon_sl = slice(
        int(np.searchsorted(lon, lon_min)),
        int(np.searchsorted(lon, lon_max, side="right")),
    )
    lat_sl = slice(
        int(np.searchsorted(lat, lat_min)),
        int(np.searchsorted(lat, lat_max, side="right")),
    )
    return lon_sl, lat_sl


def _read_time_field(
    var,
    t: int,
    lat_sl: slice,
    lon_sl: slice,
    fill: float | None,
) -> np.ndarray:
    slab = var[t]
    while slab.ndim > 2:
        slab = slab[0]
    return mask_invalid(
        np.asarray(slab[lat_sl, lon_sl], dtype=np.float32),
        fill,
    )


def load_crop(
    nc_path: Path,
    lon_sl: slice,
    lat_sl: slice,
    var_name: str,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, int]:
    """Load terrain and one time-varying field from a cropped lon/lat slice."""
    fields, extra = load_crop_multi(
        nc_path,
        lon_sl,
        lat_sl,
        [var_name],
        stride=stride,
    )
    return (
        extra.lon,
        extra.lat,
        extra.z,
        fields[var_name],
        extra.time_var,
        extra.n_time,
    )


@dataclass
class CropMeta:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    time_var: object
    n_time: int
    time_values: np.ndarray | None = None
    time_units: str | None = None
    mask: np.ndarray | None = None


def load_crop_multi(
    nc_path: Path,
    lon_sl: slice,
    lat_sl: slice,
    var_names: list[str],
    *,
    stride: int = 1,
    elev_min: float | None = None,
) -> tuple[dict[str, np.ndarray], CropMeta]:
    """Load terrain and multiple time-varying fields in one NetCDF pass."""
    st = max(1, stride)
    lon_sl_s = slice(lon_sl.start, lon_sl.stop, st)
    lat_sl_s = slice(lat_sl.start, lat_sl.stop, st)

    fields: dict[str, np.ndarray] = {}
    with nc.Dataset(nc_path, "r") as ds:
        lon = np.asarray(ds.variables["lon"][lon_sl_s], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][lat_sl_s], dtype=np.float64)

        z_var = ds.variables["z_ifc"]
        z = mask_invalid(
            np.asarray(z_var[lat_sl_s, lon_sl_s], dtype=np.float32),
            fill_value(z_var),
        )

        n_time = len(ds.dimensions["time"])
        for name in var_names:
            var = ds.variables[name]
            fill = fill_value(var)
            arr = np.empty((n_time, lat.size, lon.size), dtype=np.float32)
            for t in range(n_time):
                arr[t] = _read_time_field(var, t, lat_sl_s, lon_sl_s, fill)  # (lat, lon) order
            fields[name] = arr

        time_var = ds.variables.get("time")
        if time_var is not None:
            time_values = np.asarray(time_var[:], dtype=np.float64)
            time_units = getattr(time_var, "units", None)
        else:
            time_values = None
            time_units = None

    mask = alps_mask(z, elev_min) if elev_min is not None else None
    meta = CropMeta(
        lon=lon,
        lat=lat,
        z=z,
        time_var=time_var,
        n_time=n_time,
        time_values=time_values,
        time_units=time_units,
        mask=mask,
    )
    return fields, meta


def alps_mask(z: np.ndarray, elev_min: float) -> np.ndarray:
    return np.isfinite(z) & (z >= elev_min)


def tight_extent(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    pad_deg: float,
) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    return (
        float(lon[xs.min()]) - pad_deg,
        float(lon[xs.max()]) + pad_deg,
        float(lat[ys.min()]) - pad_deg,
        float(lat[ys.max()]) + pad_deg,
    )


def scene_centroid(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    vert_exag: float,
) -> tuple[float, float, float]:
    """Masked terrain centroid in lon/lat/Z coordinates."""
    if not np.any(mask):
        return float(lon.mean()), float(lat.mean()), float(np.nanmean(z) * vert_exag)
    lon_c = float(lon[np.any(mask, axis=0)].mean())
    lat_c = float(lat[np.any(mask, axis=1)].mean())
    z_c = float(np.nanmean(z[mask]) * vert_exag)
    return lon_c, lat_c, z_c


def estimate_crop_shape(
    nc_path: Path,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    elev_min: float = DEFAULT_ELEV_MIN,
    stride: int = 1,
) -> dict[str, int | float]:
    """Smoke-test helper: report crop dimensions and masked pixel count."""
    with nc.Dataset(nc_path, "r") as ds:
        lon_full = np.asarray(ds.variables["lon"][:])
        lat_full = np.asarray(ds.variables["lat"][:])

    lon_sl, lat_sl = crop_indices(lon_full, lat_full, lon_min, lon_max, lat_min, lat_max)
    _, meta = load_crop_multi(
        nc_path,
        lon_sl,
        lat_sl,
        [],
        stride=stride,
    )
    with nc.Dataset(nc_path, "r") as ds:
        st = max(1, stride)
        z_var = ds.variables["z_ifc"]
        z = mask_invalid(
            np.asarray(
                z_var[lat_sl.start : lat_sl.stop : st, lon_sl.start : lon_sl.stop : st],
                dtype=np.float32,
            ),
            fill_value(z_var),
        )
    mask = alps_mask(z, elev_min)
    return {
        "n_lon": meta.lon.size,
        "n_lat": meta.lat.size,
        "n_points": meta.lon.size * meta.lat.size,
        "n_masked": int(mask.sum()),
        "n_time": meta.n_time,
        "elev_min": elev_min,
    }
