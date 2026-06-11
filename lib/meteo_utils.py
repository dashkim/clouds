"""Meteorological helpers for 2D lon/lat overlay fields."""
from __future__ import annotations

import numpy as np

P0_PA = 101_325.0
SCALE_HEIGHT_M = 8500.0
# Plausible cloud-base pressure over mid-latitude terrain (Pa).
CLOUD_BASE_P_MIN = 45_000.0
CLOUD_BASE_P_MAX = 102_000.0


def sanitize_cloud_pressure(p_pa: np.ndarray) -> np.ndarray:
    """Drop missing sentinels and non-physical cloud-base pressures."""
    p = np.asarray(p_pa, dtype=np.float32).copy()
    bad = (
        ~np.isfinite(p)
        | (p <= 0.0)
        | (p < CLOUD_BASE_P_MIN)
        | (p > CLOUD_BASE_P_MAX)
        | np.isclose(p, -999.0)
    )
    p[bad] = np.nan
    return p


def pressure_to_height_msl(p_pa: np.ndarray) -> np.ndarray:
    """Convert pressure (Pa) to approximate geometric height MSL (m)."""
    p = sanitize_cloud_pressure(p_pa).astype(np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(p)
    out[valid] = (-SCALE_HEIGHT_M * np.log(p[valid] / P0_PA)).astype(np.float32)
    return out


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Horizontal wind speed (m/s) from zonal/meridional components."""
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    return np.sqrt(u_arr * u_arr + v_arr * v_arr).astype(np.float32)


def mask_cloud_overlay(
    values: np.ndarray,
    clt: np.ndarray,
    *,
    threshold: float = 5.0,
) -> np.ndarray:
    """Hide slab values where there is little or no cloud cover."""
    out = np.asarray(values, dtype=np.float32).copy()
    bad = (~np.isfinite(out)) | (clt < threshold)
    out[bad] = np.nan
    return out
