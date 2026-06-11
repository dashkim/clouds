"""Build patch-tensor datasets for lightweight inversion CNNs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib.inversion import InversionParams, cover_to_percent
from models.inversion.dataset import (
    PATCH_CHANNEL_VARS,
    _cyclical_features,
    inversion_params_from_mode,
    load_inversion_field_series,
)

PATCH_SIZE = 15
PATCH_RADIUS = PATCH_SIZE // 2

CNN_CHANNEL_NAMES = (
    "z",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "cbh",
    "t2m",
    "u10",
    "v10",
    "sp",
    "hour_sin",
    "hour_cos",
)


@dataclass
class InversionCNNDataset:
    X: np.ndarray
    y: np.ndarray
    channel_names: list[str]
    sample_time_idx: np.ndarray
    sample_lat_idx: np.ndarray
    sample_lon_idx: np.ndarray
    sample_split: np.ndarray
    times: np.ndarray
    z: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    actual_masks: np.ndarray
    hourly_actual_fraction: np.ndarray
    hourly_splits: np.ndarray
    region: str
    z_min_m: float
    inversion_params: InversionParams
    channel_medians: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray

    @property
    def n_timesteps(self) -> int:
        return self.actual_masks.shape[0]

    def ridge_mask(self) -> np.ndarray:
        return np.isfinite(self.z) & (self.z >= self.z_min_m)


def _channel_stack(fields_t: dict[str, np.ndarray], hour_sin: float, hour_cos: float) -> np.ndarray:
    """Stack (C, H, W) channels for one timestep."""
    layers: list[np.ndarray] = []
    for name in PATCH_CHANNEL_VARS:
        if name == "z":
            arr = fields_t["z"]
        elif name in ("tcc", "lcc", "mcc", "hcc"):
            arr = cover_to_percent(fields_t[name])
        else:
            arr = fields_t[name]
        layers.append(np.asarray(arr, dtype=np.float32))
    h, w = layers[0].shape
    layers.append(np.full((h, w), hour_sin, dtype=np.float32))
    layers.append(np.full((h, w), hour_cos, dtype=np.float32))
    return np.stack(layers, axis=0)


def _pad_stack(stack: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Pad (C, H, W) with per-channel medians for patch extraction."""
    c, h, w = stack.shape
    padded = np.empty((c, h + 2 * PATCH_RADIUS, w + 2 * PATCH_RADIUS), dtype=np.float32)
    for ch in range(c):
        slab = stack[ch]
        fill = medians[ch]
        finite = slab[np.isfinite(slab)]
        if finite.size:
            fill = float(np.median(finite))
        core = np.where(np.isfinite(slab), slab, fill).astype(np.float32)
        padded[ch, PATCH_RADIUS : PATCH_RADIUS + h, PATCH_RADIUS : PATCH_RADIUS + w] = core
        padded[ch, :PATCH_RADIUS, :] = fill
        padded[ch, -PATCH_RADIUS:, :] = fill
        padded[ch, :, :PATCH_RADIUS] = fill
        padded[ch, :, -PATCH_RADIUS:] = fill
    return padded


def _extract_patch(padded: np.ndarray, i: int, j: int) -> np.ndarray:
    pi = i + PATCH_RADIUS
    pj = j + PATCH_RADIUS
    return padded[
        :,
        pi - PATCH_RADIUS : pi + PATCH_RADIUS + 1,
        pj - PATCH_RADIUS : pj + PATCH_RADIUS + 1,
    ].copy()


def _stack_medians(regridded_history: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Per-channel median over all timesteps for padding."""
    n_ch = len(PATCH_CHANNEL_VARS) + 2
    samples: list[list[float]] = [[] for _ in range(n_ch)]
    for fields_t in regridded_history:
        stack = _channel_stack(fields_t, 0.0, 0.0)
        for ch in range(n_ch):
            vals = stack[ch][np.isfinite(stack[ch])]
            if vals.size:
                samples[ch].extend(vals.tolist())
    medians = np.empty(n_ch, dtype=np.float32)
    for ch in range(n_ch):
        medians[ch] = float(np.median(samples[ch])) if samples[ch] else 0.0
    medians[-2] = 0.0
    medians[-1] = 0.0
    return medians


def build_inversion_cnn_dataset(
    *,
    region: str = "west",
    z_min_m: float = 800.0,
    inversion_mode: str = "deck_only",
    inversion_params: InversionParams | None = None,
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    max_hours: int | None = None,
    patch_size: int = PATCH_SIZE,
) -> InversionCNNDataset:
    """Build 15×15 multi-channel patches for ridge cells at each hour."""
    if patch_size != PATCH_SIZE:
        raise ValueError(f"Only patch_size={PATCH_SIZE} is supported")
    params = inversion_params or inversion_params_from_mode(inversion_mode, z_min_m=z_min_m)
    series = load_inversion_field_series(
        region=region,
        inversion_params=params,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        stride=stride,
        max_hours=max_hours,
    )

    channel_medians = _stack_medians(series.regridded_history)
    n_time = series.n_timesteps
    n_ridge = series.ridge_lat_idx.size
    n_ch = len(CNN_CHANNEL_NAMES)
    n_samples = n_time * n_ridge
    X = np.empty((n_samples, n_ch, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    sample_time_idx = np.empty(n_samples, dtype=np.int32)
    sample_lat_idx = np.empty(n_samples, dtype=np.int32)
    sample_lon_idx = np.empty(n_samples, dtype=np.int32)
    sample_split = np.empty(n_samples, dtype=np.int8)

    row = 0
    for t in range(n_time):
        ts = pd.Timestamp(series.times[t])
        hour_sin, hour_cos, _, _ = _cyclical_features(ts)
        stack = _channel_stack(series.regridded_history[t], hour_sin, hour_cos)
        padded = _pad_stack(stack, channel_medians)

        for k in range(n_ridge):
            i = int(series.ridge_lat_idx[k])
            j = int(series.ridge_lon_idx[k])
            inv_val = series.actual_masks[t, i, j]
            if not np.isfinite(inv_val):
                continue
            X[row] = _extract_patch(padded, i, j)
            y[row] = float(inv_val)
            sample_time_idx[row] = t
            sample_lat_idx[row] = i
            sample_lon_idx[row] = j
            sample_split[row] = series.hourly_splits[t]
            row += 1

    if row < n_samples:
        X = X[:row]
        y = y[:row]
        sample_time_idx = sample_time_idx[:row]
        sample_lat_idx = sample_lat_idx[:row]
        sample_lon_idx = sample_lon_idx[:row]
        sample_split = sample_split[:row]

    channel_mean = np.nanmean(X, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    channel_std = np.nanstd(X, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    channel_std[channel_std < 1e-6] = 1.0
    X = ((X - channel_mean[:, None, None]) / channel_std[:, None, None]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return InversionCNNDataset(
        X=X,
        y=y,
        channel_names=list(CNN_CHANNEL_NAMES),
        sample_time_idx=sample_time_idx,
        sample_lat_idx=sample_lat_idx,
        sample_lon_idx=sample_lon_idx,
        sample_split=sample_split,
        times=series.times,
        z=series.z,
        lon=series.lon,
        lat=series.lat,
        actual_masks=series.actual_masks,
        hourly_actual_fraction=series.hourly_actual_fraction,
        hourly_splits=series.hourly_splits,
        region=region,
        z_min_m=params.z_min_m,
        inversion_params=params,
        channel_medians=channel_medians,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
