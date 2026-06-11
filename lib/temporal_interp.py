"""Temporal upsampling for patch field time series (movie smoothing)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from lib.inversion import detect_inversion_mask
from lib.terrain_mesh import PatchGrid


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)


def lerp_time_series(series: np.ndarray, time_idx: float) -> np.ndarray:
    """Linearly interpolate a (time, lat, lon) stack at fractional time_idx."""
    n = series.shape[0]
    if n == 0:
        raise ValueError("empty time series")
    t = max(0.0, min(float(time_idx), n - 1))
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, n - 1)
    alpha = t - i0
    if i0 == i1 or alpha < 1e-6:
        return series[i0]
    return _lerp(series[i0], series[i1], alpha)


def timestamp_at_index(times: list[str] | None, time_idx: float) -> str | None:
    if not times:
        return None
    n = len(times)
    t = max(0.0, min(float(time_idx), n - 1))
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, n - 1)
    alpha = t - i0
    if i0 == i1 or alpha < 1e-6:
        return times[i0]
    return _interp_timestamps(times[i0], times[i1], alpha)


def _interp_timestamps(t0: str, t1: str, alpha: float) -> str:
    a = pd.Timestamp(t0)
    b = pd.Timestamp(t1)
    mid = a + (b - a) * alpha
    return str(mid)


def densify_patch_grid(grid: PatchGrid, steps_per_interval: int) -> PatchGrid:
    """
    Insert blended transition frames between consecutive hourly timesteps.

    steps_per_interval=4 yields 4 sub-frames per hour gap (3 blends + endpoint).
    Total frames: (T - 1) * steps_per_interval + 1.
    """
    if steps_per_interval <= 1 or grid.overlay is None:
        return grid

    n_src = grid.overlay.shape[0]
    if n_src < 2:
        return grid

    def expand_series(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None or arr.ndim != 3 or arr.shape[0] != n_src:
            return arr
        n_lat, n_lon = arr.shape[1], arr.shape[2]
        n_out = (n_src - 1) * steps_per_interval + 1
        out = np.empty((n_out, n_lat, n_lon), dtype=np.float32)
        out_idx = 0
        for t in range(n_src - 1):
            out[out_idx] = arr[t]
            out_idx += 1
            for s in range(1, steps_per_interval):
                alpha = s / steps_per_interval
                out[out_idx] = _lerp(arr[t], arr[t + 1], alpha)
                out_idx += 1
        out[out_idx] = arr[n_src - 1]
        return out

    frame_times: list[str] | None = None
    if grid.era5_frame_times and len(grid.era5_frame_times) == n_src:
        frame_times = []
        for t in range(n_src - 1):
            frame_times.append(grid.era5_frame_times[t])
            for s in range(1, steps_per_interval):
                alpha = s / steps_per_interval
                frame_times.append(
                    _interp_timestamps(
                        grid.era5_frame_times[t],
                        grid.era5_frame_times[t + 1],
                        alpha,
                    )
                )
        frame_times.append(grid.era5_frame_times[-1])

    clt_out = expand_series(grid.clt)
    low_out = expand_series(grid.cloud_deck_cover)
    med_out = expand_series(grid.medium_cover) if hasattr(grid, "medium_cover") else None
    base_out = expand_series(grid.cloud_base_m)

    inversion_out = None
    if clt_out is not None and grid.slab_mode == "inversion":
        n_out = clt_out.shape[0]
        inversion_out = np.empty((n_out, grid.z.shape[0], grid.z.shape[1]), dtype=np.float32)
        for t in range(n_out):
            inversion_out[t] = detect_inversion_mask(
                grid.z,
                cloud_cover=clt_out[t],
                low_cover=None if low_out is None else low_out[t],
                medium_cover=None if med_out is None else med_out[t],
                cloud_base_m=None if base_out is None else base_out[t],
            )
    elif grid.slab_mode == "inversion_compare":
        compare_src = grid.comparison_field if grid.comparison_field is not None else grid.overlay
        inversion_out = expand_series(compare_src)
    elif grid.slab_mode == "feature_overlay":
        inversion_out = expand_series(grid.inversion)
    else:
        inversion_out = expand_series(grid.inversion)

    overlay_out = inversion_out
    if grid.slab_mode == "inversion_compare":
        compare_src = grid.comparison_field if grid.comparison_field is not None else grid.overlay
        overlay_out = expand_series(compare_src)
    elif inversion_out is not None:
        overlay_out = inversion_out
    else:
        overlay_out = expand_series(grid.overlay)

    comparison_out = expand_series(grid.comparison_field) if grid.comparison_field is not None else None
    predicted_out = expand_series(grid.predicted_inversion) if grid.predicted_inversion is not None else None
    inversion_src = expand_series(grid.inversion) if grid.inversion is not None else None
    t2m_out = expand_series(grid.feature_t2m) if grid.feature_t2m is not None else None
    u_out = expand_series(grid.u_wind) if grid.u_wind is not None else None
    v_out = expand_series(grid.v_wind) if grid.v_wind is not None else None

    return replace(
        grid,
        overlay=overlay_out,
        clt=clt_out,
        inversion=inversion_src if inversion_src is not None else inversion_out,
        cloud_base_m=base_out,
        cloud_deck_cover=low_out,
        medium_cover=med_out,
        comparison_field=comparison_out,
        predicted_inversion=predicted_out,
        feature_t2m=t2m_out,
        u_wind=u_out,
        v_wind=v_out,
        era5_frame_times=frame_times,
    )
