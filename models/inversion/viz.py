"""Helpers for ML-driven visualization overlays."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lib.alps_region import crop_indices, default_patch_stride, resolve_patch_bounds
from lib.era5_io import (
    era5_to_icon_grid,
    iso_to_datetime,
    load_era5_hourly_fields,
    nearest_era5_index,
)
from lib.inversion import cover_to_percent
from lib.patch_fields import _load_icon_terrain
from lib.paths import find_era5_grib, find_lonlat_nc

import netCDF4 as nc

SPATIAL_FEATURE_NAMES = frozenset({
    "z", "tcc", "lcc", "mcc", "hcc", "cbh", "t2m", "u10", "v10", "sp",
})

FEATURE_LABELS = {
    "lcc": "Low cloud cover (LCC)",
    "tcc": "Total cloud cover",
    "t2m": "2 m temperature",
    "u10": "10 m zonal wind",
    "v10": "10 m meridional wind",
    "cbh": "Cloud base height",
    "z": "Terrain elevation",
}


def load_ridge_meta(model_path: Path) -> dict:
    payload = joblib.load(model_path)
    return payload["meta"]


def top_spatial_drivers(model_path: Path, *, n: int = 3) -> list[tuple[str, float]]:
    """Return top spatial features by |effective Ridge coefficient|."""
    payload = joblib.load(model_path)
    pipe = payload["pipeline"]
    names = payload["meta"]["feature_names"]
    coef = pipe.named_steps["ridge"].coef_
    scale = pipe.named_steps["scaler"].scale_
    eff = coef / scale
    ranked: list[tuple[str, float]] = []
    for name, weight in zip(names, eff):
        if name in SPATIAL_FEATURE_NAMES:
            ranked.append((name, float(weight)))
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return ranked[:n]


def format_driver_notes(
    drivers: list[tuple[str, float]],
    *,
    patch_means: dict[str, float] | None = None,
) -> str:
    """One-line summary of model spatial drivers for frame annotation."""
    parts: list[str] = []
    for name, weight in drivers:
        label = FEATURE_LABELS.get(name, name)
        sign = "+" if weight >= 0 else "−"
        text = f"{label} ({sign}|w|)"
        if patch_means and name in patch_means:
            val = patch_means[name]
            if name in ("lcc", "tcc", "mcc", "hcc"):
                text += f" μ={val:.0f}%"
            elif name == "t2m":
                text += f" μ={val:.1f} K"
            elif name in ("u10", "v10"):
                text += f" μ={val:.1f} m/s"
        parts.append(text)
    return "  ·  ".join(parts)


def patch_mean_features(
    *,
    lcc: np.ndarray,
    t2m: np.ndarray | None,
    u10: np.ndarray,
    v10: np.ndarray,
) -> dict[str, float]:
    """Patch-mean values for annotation (single timestep)."""
    out: dict[str, float] = {}
    lcc_pct = cover_to_percent(lcc)
    valid = lcc_pct[np.isfinite(lcc_pct)]
    out["lcc"] = float(np.nanmean(valid)) if valid.size else 0.0
    if t2m is not None:
        t_valid = t2m[np.isfinite(t2m)]
        out["t2m"] = float(np.nanmean(t_valid)) if t_valid.size else 0.0
    u_valid = u10[np.isfinite(u10)]
    v_valid = v10[np.isfinite(v10)]
    out["u10"] = float(np.nanmean(u_valid)) if u_valid.size else 0.0
    out["v10"] = float(np.nanmean(v_valid)) if v_valid.size else 0.0
    return out


def load_era5_gridded_features(
    *,
    region: str = "east_core",
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    era5_start: str,
    era5_n_frames: int,
    variables: tuple[str, ...] = ("t2m", "u10", "v10"),
) -> dict[str, np.ndarray]:
    """Load ERA5 fields on the ICON patch grid for a fixed hourly window."""
    icon_path = icon_nc or find_lonlat_nc()
    grib_path = era5_grib or find_era5_grib()
    stride = stride if stride is not None else default_patch_stride(region)

    lon_min, lon_max, lat_min, lat_max, _, _ = resolve_patch_bounds(region=region)
    with nc.Dataset(icon_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            lon_min, lon_max, lat_min, lat_max,
        )
    lon, lat, _z = _load_icon_terrain(icon_path, lon_sl, lat_sl, stride)

    ds_hourly = load_era5_hourly_fields(
        grib_path,
        lon_min=lon_min - 0.5,
        lon_max=lon_max + 0.5,
        lat_min=lat_min - 0.5,
        lat_max=lat_max + 0.5,
        variables=variables,
    )
    era5_times = pd.to_datetime(ds_hourly["time"].values)
    start_idx = nearest_era5_index(era5_times.to_numpy(), iso_to_datetime(era5_start))
    indices = list(range(start_idx, min(start_idx + era5_n_frames, len(era5_times))))
    n_time = len(indices)

    era5_lat = np.asarray(ds_hourly["latitude"].values, dtype=np.float64)
    era5_lon = np.asarray(ds_hourly["longitude"].values, dtype=np.float64)

    out: dict[str, np.ndarray] = {}
    for var in variables:
        if var not in ds_hourly:
            raise KeyError(f"ERA5 variable {var!r} not in dataset")
        native = np.asarray(ds_hourly[var].values, dtype=np.float32)
        stack = np.empty((n_time, lat.size, lon.size), dtype=np.float32)
        for out_t, era5_t in enumerate(indices):
            stack[out_t] = era5_to_icon_grid(native[era5_t], era5_lat, era5_lon, lat, lon)
        out[var] = stack
    return out
