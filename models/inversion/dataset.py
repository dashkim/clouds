"""Shared inversion field loading, Ridge tabular datasets, and ML metrics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from lib.alps_region import crop_indices, default_patch_stride, resolve_patch_bounds
from lib.era5_io import (
    cbh_at_index,
    era5_to_icon_grid,
    load_era5_hourly_fields,
    preload_cbh_series,
    step_dataset,
)
from lib.inversion import (
    InversionParams,
    cover_to_percent,
    detect_inversion_mask,
    ridge_inversion_fraction,
)
from lib.patch_fields import _load_icon_terrain
from lib.paths import find_era5_grib, find_lonlat_nc

import netCDF4 as nc

ML_HOURLY_VARS = ("tcc", "lcc", "mcc", "hcc", "t2m", "u10", "v10", "sp")
PER_CELL_FEATURES = (
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
    "doy_sin",
    "doy_cos",
    "lcc_mean_lag1",
    "mcc_mean_lag1",
    "tcc_mean_lag1",
    "lcc_mean_lag2",
    "mcc_mean_lag2",
    "tcc_mean_lag2",
    "lcc_mean_lag3",
    "mcc_mean_lag3",
    "tcc_mean_lag3",
)

PATCH_CHANNEL_VARS = ("z", "tcc", "lcc", "mcc", "hcc", "cbh", "t2m", "u10", "v10", "sp")


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2


SPLIT_NAMES = ("train", "val", "test")


def time_split_label(ts: pd.Timestamp) -> Split:
    """Jan–Feb train, Mar val, Apr+ test."""
    month = int(ts.month)
    if month <= 2:
        return Split.TRAIN
    if month == 3:
        return Split.VAL
    return Split.TEST


def inversion_params_from_mode(
    mode: str,
    *,
    z_min_m: float = 800.0,
    cbh_margin_m: float = 100.0,
) -> InversionParams:
    if mode not in ("deck_only", "phenomenological"):
        raise ValueError(f"Unknown inversion mode {mode!r}")
    return InversionParams(mode=mode, z_min_m=z_min_m, cbh_margin_m=cbh_margin_m)


def _cyclical_features(ts: pd.Timestamp) -> tuple[float, float, float, float]:
    hour = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear
    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    doy_sin = float(np.sin(2.0 * np.pi * doy / 365.25))
    doy_cos = float(np.cos(2.0 * np.pi * doy / 365.25))
    return hour_sin, hour_cos, doy_sin, doy_cos


def _patch_means(fields: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("tcc", "lcc", "mcc"):
        arr = cover_to_percent(fields[key])
        valid = arr[np.isfinite(arr)]
        out[key] = float(np.nanmean(valid)) if valid.size else 0.0
    return out


@dataclass
class InversionFieldSeries:
    """Shared hourly ERA5 fields and inversion masks for a patch region."""

    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    times: np.ndarray
    actual_masks: np.ndarray
    hourly_actual_fraction: np.ndarray
    hourly_splits: np.ndarray
    regridded_history: list[dict[str, np.ndarray]]
    patch_mean_history: list[dict[str, float]]
    region: str
    inversion_params: InversionParams
    ridge_lat_idx: np.ndarray
    ridge_lon_idx: np.ndarray

    @property
    def n_timesteps(self) -> int:
        return self.actual_masks.shape[0]

    @property
    def z_min_m(self) -> float:
        return self.inversion_params.z_min_m

    def ridge_mask(self) -> np.ndarray:
        return np.isfinite(self.z) & (self.z >= self.z_min_m)


def load_inversion_field_series(
    *,
    region: str = "east_core",
    inversion_params: InversionParams | None = None,
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    max_hours: int | None = None,
) -> InversionFieldSeries:
    """Load terrain, hourly ERA5 fields regridded to ICON, and inversion masks."""
    params = inversion_params or InversionParams()
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
    lon, lat, z = _load_icon_terrain(icon_path, lon_sl, lat_sl, stride)
    ridge = np.isfinite(z) & (z >= params.z_min_m)
    ridge_lat_idx, ridge_lon_idx = np.where(ridge)
    if ridge_lat_idx.size == 0:
        raise ValueError(f"No ridge cells at z>={params.z_min_m} m in region {region!r}")

    ds_hourly = load_era5_hourly_fields(
        grib_path,
        lon_min=lon_min - 0.5,
        lon_max=lon_max + 0.5,
        lat_min=lat_min - 0.5,
        lat_max=lat_max + 0.5,
        variables=ML_HOURLY_VARS,
    )
    preload_cbh_series(grib_path)
    era5_times = pd.to_datetime(ds_hourly["time"].values)
    n_time = len(era5_times)
    if max_hours is not None:
        n_time = min(n_time, max_hours)

    era5_lat = np.asarray(ds_hourly["latitude"].values, dtype=np.float64)
    era5_lon = np.asarray(ds_hourly["longitude"].values, dtype=np.float64)
    hourly_native: dict[str, np.ndarray] = {}
    for var in ML_HOURLY_VARS:
        if var in ds_hourly:
            hourly_native[var] = np.asarray(ds_hourly[var].values, dtype=np.float32)

    ds_step = step_dataset(grib_path)
    cbh_lat = np.asarray(ds_step["latitude"].values, dtype=np.float64)
    cbh_lon = np.asarray(ds_step["longitude"].values, dtype=np.float64)

    n_lat, n_lon = z.shape
    actual_masks = np.empty((n_time, n_lat, n_lon), dtype=np.float32)
    hourly_actual_fraction = np.empty(n_time, dtype=np.float32)
    hourly_splits = np.empty(n_time, dtype=np.int8)

    patch_mean_history: list[dict[str, float]] = []
    regridded_history: list[dict[str, np.ndarray]] = []

    for t in range(n_time):
        ts = pd.Timestamp(era5_times[t])
        hourly_splits[t] = int(time_split_label(ts))

        fields_t: dict[str, np.ndarray] = {}
        for var in ML_HOURLY_VARS:
            if var in hourly_native:
                fields_t[var] = era5_to_icon_grid(
                    hourly_native[var][t], era5_lat, era5_lon, lat, lon,
                )
        cbh = era5_to_icon_grid(
            cbh_at_index(grib_path, t, era5_times.to_numpy()),
            cbh_lat, cbh_lon, lat, lon,
        )
        fields_t["cbh"] = cbh
        fields_t["z"] = z

        mask = detect_inversion_mask(
            z,
            cloud_cover=fields_t["tcc"],
            low_cover=fields_t["lcc"],
            medium_cover=fields_t["mcc"],
            cloud_base_m=cbh,
            params=params,
        )
        actual_masks[t] = mask
        hourly_actual_fraction[t] = ridge_inversion_fraction(
            mask, z, z_min_m=params.z_min_m,
        )

        patch_mean_history.append(_patch_means(fields_t))
        regridded_history.append(fields_t)

    return InversionFieldSeries(
        lon=lon,
        lat=lat,
        z=z,
        times=era5_times[:n_time].to_numpy(),
        actual_masks=actual_masks,
        hourly_actual_fraction=hourly_actual_fraction,
        hourly_splits=hourly_splits,
        regridded_history=regridded_history,
        patch_mean_history=patch_mean_history,
        region=region,
        inversion_params=params,
        ridge_lat_idx=ridge_lat_idx,
        ridge_lon_idx=ridge_lon_idx,
    )


@dataclass
class InversionMLDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
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

    @property
    def n_timesteps(self) -> int:
        return self.actual_masks.shape[0]

    def ridge_mask(self) -> np.ndarray:
        return np.isfinite(self.z) & (self.z >= self.z_min_m)


def build_inversion_ml_dataset(
    *,
    region: str = "east_core",
    z_min_m: float = 800.0,
    inversion_mode: str = "phenomenological",
    inversion_params: InversionParams | None = None,
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    max_hours: int | None = None,
) -> InversionMLDataset:
    """Build ridge-cell samples and full-hour actual inversion masks."""
    params = inversion_params or inversion_params_from_mode(inversion_mode, z_min_m=z_min_m)
    series = load_inversion_field_series(
        region=region,
        inversion_params=params,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        stride=stride,
        max_hours=max_hours,
    )

    n_time = series.n_timesteps
    n_ridge = series.ridge_lat_idx.size
    n_features = len(PER_CELL_FEATURES)
    n_samples = n_time * n_ridge
    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    sample_time_idx = np.empty(n_samples, dtype=np.int32)
    sample_lat_idx = np.empty(n_samples, dtype=np.int32)
    sample_lon_idx = np.empty(n_samples, dtype=np.int32)
    sample_split = np.empty(n_samples, dtype=np.int8)

    row = 0
    for t in range(n_time):
        ts = pd.Timestamp(series.times[t])
        hour_sin, hour_cos, doy_sin, doy_cos = _cyclical_features(ts)
        fields_t = series.regridded_history[t]

        lag_feats: dict[str, float] = {}
        for lag in (1, 2, 3):
            if t >= lag:
                pm = series.patch_mean_history[t - lag]
            else:
                pm = series.patch_mean_history[0]
            lag_feats[f"lcc_mean_lag{lag}"] = pm["lcc"]
            lag_feats[f"mcc_mean_lag{lag}"] = pm["mcc"]
            lag_feats[f"tcc_mean_lag{lag}"] = pm["tcc"]

        for k in range(n_ridge):
            i = int(series.ridge_lat_idx[k])
            j = int(series.ridge_lon_idx[k])
            inv_val = series.actual_masks[t, i, j]
            if not np.isfinite(inv_val):
                continue

            feat_row = [
                float(series.z[i, j]),
                float(cover_to_percent(fields_t["tcc"][i, j])),
                float(cover_to_percent(fields_t["lcc"][i, j])),
                float(cover_to_percent(fields_t["mcc"][i, j])),
                float(cover_to_percent(fields_t["hcc"][i, j])),
                float(fields_t["cbh"][i, j]),
                float(fields_t["t2m"][i, j]),
                float(fields_t["u10"][i, j]),
                float(fields_t["v10"][i, j]),
                float(fields_t["sp"][i, j]),
                hour_sin,
                hour_cos,
                doy_sin,
                doy_cos,
                lag_feats["lcc_mean_lag1"],
                lag_feats["mcc_mean_lag1"],
                lag_feats["tcc_mean_lag1"],
                lag_feats["lcc_mean_lag2"],
                lag_feats["mcc_mean_lag2"],
                lag_feats["tcc_mean_lag2"],
                lag_feats["lcc_mean_lag3"],
                lag_feats["mcc_mean_lag3"],
                lag_feats["tcc_mean_lag3"],
            ]
            X[row] = feat_row
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

    return InversionMLDataset(
        X=X,
        y=y,
        feature_names=list(PER_CELL_FEATURES),
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
    )


def masks_to_fraction(masks: np.ndarray, z: np.ndarray, z_min_m: float) -> np.ndarray:
    """Mean inversion value over ridge cells per timestep."""
    ridge = np.isfinite(z) & (z >= z_min_m)
    if not np.any(ridge):
        return np.zeros(masks.shape[0], dtype=np.float32)
    out = np.empty(masks.shape[0], dtype=np.float32)
    for t in range(masks.shape[0]):
        slab = masks[t]
        valid = ridge & np.isfinite(slab)
        if not np.any(valid):
            out[t] = 0.0
        else:
            out[t] = float(np.mean(slab[valid] >= 0.5))
    return out


def build_agreement_map(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    z: np.ndarray,
    z_min_m: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Categorical agreement on ridge cells.

    0=background, 1=TP (green), 2=FP (red), 3=FN (blue).
    """
    ridge = np.isfinite(z) & (z >= z_min_m)
    act = np.asarray(actual, dtype=np.float32) >= threshold
    pred = np.clip(np.asarray(predicted, dtype=np.float32), 0.0, 1.0) >= threshold
    out = np.zeros(actual.shape, dtype=np.float32)
    out[ridge & act & pred] = 1.0
    out[ridge & (~act) & pred] = 2.0
    out[ridge & act & (~pred)] = 3.0
    out[~ridge] = np.nan
    return out


def pick_best_test_day(
    times: np.ndarray,
    actual_fraction: np.ndarray,
    splits: np.ndarray,
    *,
    min_fraction: float = 0.05,
) -> tuple[str, int]:
    """Return ISO start hour and frame count for strongest test inversion day."""
    ts = pd.to_datetime(times)
    test_mask = splits == int(Split.TEST)
    df = pd.DataFrame({
        "time": ts,
        "fraction": actual_fraction,
        "date": ts.date,
        "test": test_mask,
    })
    test_df = df[df["test"] & (df["fraction"] >= min_fraction)]
    if test_df.empty:
        test_df = df[df["test"]]
    if test_df.empty:
        val_df = df[df["fraction"] >= min_fraction]
        if val_df.empty:
            val_df = df
        test_df = val_df

    daily = test_df.groupby("date")["fraction"].mean().sort_values(ascending=False)
    best_date = daily.index[0]
    day_rows = test_df[test_df["date"] == best_date].sort_values("time")
    start = pd.Timestamp(day_rows["time"].iloc[0])
    n_frames = min(12, len(day_rows))
    return start.strftime("%Y-%m-%dT%H:%M"), n_frames
