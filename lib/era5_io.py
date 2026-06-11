"""ERA5 GRIB loading, cropping, regridding, and time alignment."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# cfgrib dataset group indices in 2013_germany.grib
GROUP_HOURLY = 1
GROUP_STEP = 2

# Canonical names → possible GRIB/cfgrib variable names
VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tcc": ("tcc",),
    "lcc": ("lcc",),
    "mcc": ("mcc",),
    "hcc": ("hcc",),
    "t2m": ("t2m", "2t"),
    "u10": ("u10", "10u"),
    "v10": ("v10", "10v"),
    "sp": ("sp",),
    "cbh": ("cbh",),
    "tp": ("tp",),
    "cp": ("cp",),
    "e": ("e",),
}

INVERSION_VARS = ("tcc", "lcc", "mcc", "cbh")
ML_HOURLY_VARS = ("tcc", "lcc", "mcc", "hcc", "t2m", "u10", "v10", "sp")
_ML_HOURLY_SHORT_NAMES = ("tcc", "lcc", "mcc", "hcc", "2t", "10u", "10v", "sp")


_GROUPS_CACHE: dict[str, list[xr.Dataset]] = {}
_HOURLY_CACHE: dict[str, xr.Dataset] = {}
_ML_HOURLY_CACHE: dict[str, xr.Dataset] = {}
_STEP_CACHE: dict[str, xr.Dataset] = {}


def open_era5_groups(grib_path: Path | str) -> list[xr.Dataset]:
    key = str(Path(grib_path).resolve())
    if key not in _GROUPS_CACHE:
        _GROUPS_CACHE[key] = list(cfgrib.open_datasets(key))
    return _GROUPS_CACHE[key]


def _open_by_short_name(grib_path: Path | str, short_name: str) -> xr.Dataset:
    return xr.open_dataset(
        str(grib_path),
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}},
    )


def _resolve_var(ds: xr.Dataset, canonical: str) -> str | None:
    for name in VAR_ALIASES.get(canonical, (canonical,)):
        if name in ds.data_vars:
            return name
    return None


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def hourly_dataset(grib_path: Path | str) -> xr.Dataset:
    """Hourly instantaneous fields (tcc, lcc, mcc) — loaded via shortName filters."""
    key = str(Path(grib_path).resolve())
    if key not in _HOURLY_CACHE:
        parts = [_open_by_short_name(key, sn) for sn in ("tcc", "lcc", "mcc")]
        merged = xr.merge(parts, compat="override")
        _HOURLY_CACHE[key] = _normalize_coords(merged)
    return _HOURLY_CACHE[key]


def ml_hourly_dataset(grib_path: Path | str) -> xr.Dataset:
    """Hourly fields for ML features (cloud, surface met, winds)."""
    key = str(Path(grib_path).resolve())
    if key not in _ML_HOURLY_CACHE:
        parts = [_open_by_short_name(key, sn) for sn in _ML_HOURLY_SHORT_NAMES]
        merged = xr.merge(parts, compat="override")
        _ML_HOURLY_CACHE[key] = _normalize_coords(merged)
    return _ML_HOURLY_CACHE[key]


def step_dataset(grib_path: Path | str) -> xr.Dataset:
    """Fields on (time, step) including cbh."""
    key = str(Path(grib_path).resolve())
    if key not in _STEP_CACHE:
        _STEP_CACHE[key] = _normalize_coords(_open_by_short_name(key, "cbh"))
    return _STEP_CACHE[key]


def era5_time_values(ds: xr.Dataset) -> np.ndarray:
    if "valid_time" in ds.coords:
        vt = ds["valid_time"].values
        if vt.ndim == 2:
            return pd.to_datetime(vt.ravel()).to_numpy()
        return pd.to_datetime(vt).to_numpy()
    if "time" in ds.coords:
        return pd.to_datetime(ds["time"].values).to_numpy()
    raise KeyError("No time or valid_time coordinate in ERA5 dataset")


def crop_era5(
    ds: xr.Dataset,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.Dataset:
    lat = ds["latitude"].values
    if lat[0] > lat[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    return ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=lat_slice,
    )


def era5_to_icon_grid(
    field: np.ndarray,
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
    icon_lat: np.ndarray,
    icon_lon: np.ndarray,
) -> np.ndarray:
    """Bilinear regrid ERA5 2D field onto ICON 1D lon/lat mesh."""
    lat = np.asarray(era5_lat, dtype=np.float64)
    lon = np.asarray(era5_lon, dtype=np.float64)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]

    interp = RegularGridInterpolator(
        (lat, lon),
        np.asarray(field, dtype=np.float64),
        bounds_error=False,
        fill_value=np.nan,
    )
    lat_2d, lon_2d = np.meshgrid(icon_lat, icon_lon, indexing="ij")
    pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
    out = interp(pts).reshape(lat_2d.shape).astype(np.float32)
    return out


def _cbh_lookup(ds_step: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (time, step) valid_time and cbh for nearest-time lookup."""
    vt = pd.to_datetime(ds_step["valid_time"].values.ravel())
    cbh_name = _resolve_var(ds_step, "cbh")
    if cbh_name is None:
        raise KeyError("cbh not found in ERA5 step group")
    cbh = np.asarray(ds_step[cbh_name].values, dtype=np.float32)
    cbh_flat = cbh.reshape(-1, cbh.shape[-2], cbh.shape[-1])
    return vt.to_numpy(), cbh_flat


def cbh_at_time(ds_step: xr.Dataset, target: datetime | pd.Timestamp) -> np.ndarray:
    """Single cbh field for nearest valid_time."""
    vt, cbh_flat = _cbh_lookup(ds_step)
    target_ts = pd.Timestamp(target)
    if target_ts.tzinfo is not None:
        target_ts = target_ts.tz_convert("UTC").tz_localize(None)
    idx = int(np.argmin(np.abs(vt - target_ts.to_numpy())))
    return cbh_flat[idx]


def nearest_era5_index(times: np.ndarray, target: datetime | pd.Timestamp) -> int:
    ts = pd.to_datetime(times)
    target_ts = pd.Timestamp(target)
    if target_ts.tzinfo is not None:
        target_ts = target_ts.tz_convert("UTC").tz_localize(None)
    return int(np.argmin(np.abs(ts - target_ts.to_numpy())))


_CBH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def preload_cbh_series(grib_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Cache flattened valid_time and cbh arrays from the step group."""
    key = str(Path(grib_path).resolve())
    if key not in _CBH_CACHE:
        _CBH_CACHE[key] = _cbh_lookup(step_dataset(grib_path))
    return _CBH_CACHE[key]


def cbh_at_index(grib_path: Path | str, time_idx: int, era5_times: np.ndarray) -> np.ndarray:
    """cbh field for the hourly time index (nearest step-group valid_time)."""
    vt_flat, cbh_flat = preload_cbh_series(grib_path)
    target = pd.Timestamp(era5_times[time_idx])
    idx = int(np.argmin(np.abs(vt_flat - target.to_numpy())))
    return cbh_flat[idx]


def load_era5_hourly_fields(
    grib_path: Path | str,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    variables: tuple[str, ...] = INVERSION_VARS,
) -> xr.Dataset:
    base = (
        ml_hourly_dataset(grib_path)
        if set(variables) - {"cbh"} - set(INVERSION_VARS)
        else hourly_dataset(grib_path)
    )
    ds = crop_era5(base, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    keep: dict[str, Any] = {}
    for canonical in variables:
        if canonical == "cbh":
            continue
        name = _resolve_var(ds, canonical)
        if name is not None:
            keep[canonical] = ds[name]
    out = xr.Dataset(keep)
    if "time" in ds.coords:
        out = out.assign_coords(time=ds["time"])
    if "latitude" in ds.coords:
        out = out.assign_coords(latitude=ds["latitude"], longitude=ds["longitude"])
    return out


def regrid_era5_field_to_patch(
    da: xr.DataArray,
    icon_lat: np.ndarray,
    icon_lon: np.ndarray,
    *,
    time_idx: int | None = None,
) -> np.ndarray:
    if time_idx is not None and "time" in da.dims:
        field = np.asarray(da.isel(time=time_idx).values, dtype=np.float32)
    else:
        field = np.asarray(da.values, dtype=np.float32)
    era5_lat = np.asarray(da["latitude"].values, dtype=np.float64)
    era5_lon = np.asarray(da["longitude"].values, dtype=np.float64)
    return era5_to_icon_grid(field, era5_lat, era5_lon, icon_lat, icon_lon)


def _inventory_grib_eccodes(grib_path: Path | str) -> list[dict[str, Any]]:
    """Fast GRIB header scan via eccodes (no array load)."""
    import eccodes

    path = Path(grib_path)
    groups: dict[tuple, dict[str, Any]] = {}

    with path.open("rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                short_name = eccodes.codes_get(gid, "shortName")
                units = eccodes.codes_get(gid, "units")
                level = eccodes.codes_get(gid, "level")
                typ = eccodes.codes_get(gid, "typeOfLevel")
                date = eccodes.codes_get(gid, "date")
                time_val = eccodes.codes_get(gid, "time")
                step = eccodes.codes_get(gid, "step")
                nlat = eccodes.codes_get(gid, "Nj")
                nlon = eccodes.codes_get(gid, "Ni")
                lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lat2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                lon2 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
                valid = eccodes.codes_get(gid, "validityDate")
                valid_t = eccodes.codes_get(gid, "validityTime")
            except Exception:
                eccodes.codes_release(gid)
                continue

            key = (typ, int(level), int(step) if step else 0)
            if key not in groups:
                groups[key] = {
                    "typeOfLevel": typ,
                    "level": level,
                    "step": step,
                    "nlat": nlat,
                    "nlon": nlon,
                    "lat_range": (lat1, lat2),
                    "lon_range": (lon1, lon2),
                    "dates": set(),
                    "valid_times": set(),
                    "variables": {},
                }
            g = groups[key]
            g["dates"].add((date, time_val))
            g["valid_times"].add((valid, valid_t))
            g["variables"][short_name] = {"shortName": short_name, "units": units}

            eccodes.codes_release(gid)

    rows: list[dict[str, Any]] = []
    for i, g in enumerate(groups.values()):
        valid_sorted = sorted(g["valid_times"])
        t0 = f"{valid_sorted[0][0]:08d} {valid_sorted[0][1]:04d}" if valid_sorted else None
        t1 = f"{valid_sorted[-1][0]:08d} {valid_sorted[-1][1]:04d}" if valid_sorted else None
        lat1, lat2 = g["lat_range"]
        lon1, lon2 = g["lon_range"]
        dlat = abs(lat1 - lat2) / max(g["nlat"] - 1, 1)
        dlon = abs(lon1 - lon2) / max(g["nlon"] - 1, 1)
        vars_info = [
            {"name": sn, "shortName": info["shortName"], "units": info["units"], "dims": []}
            for sn, info in sorted(g["variables"].items())
        ]
        rows.append({
            "group": i,
            "sizes": {"latitude": g["nlat"], "longitude": g["nlon"], "n_valid": len(valid_sorted)},
            "time_start": t0,
            "time_end": t1,
            "n_time": len(valid_sorted),
            "dlat": round(dlat, 3),
            "dlon": round(dlon, 3),
            "variables": vars_info,
            "typeOfLevel": g["typeOfLevel"],
            "step": g["step"],
        })
    return rows


def inventory_grib(grib_path: Path | str) -> list[dict[str, Any]]:
    """Summarize GRIB contents (fast eccodes header scan)."""
    try:
        return _inventory_grib_eccodes(grib_path)
    except Exception:
        rows: list[dict[str, Any]] = []
        for i, ds in enumerate(open_era5_groups(grib_path)):
            times = era5_time_values(ds) if "time" in ds.coords or "valid_time" in ds.coords else []
            t0 = str(times[0]) if len(times) else None
            t1 = str(times[-1]) if len(times) else None
            vars_info = []
            for v in ds.data_vars:
                da = ds[v]
                vars_info.append({
                    "name": v,
                    "shortName": da.attrs.get("GRIB_shortName", v),
                    "units": da.attrs.get("units"),
                    "dims": list(da.dims),
                })
            lat = ds["latitude"].values if "latitude" in ds.coords else []
            lon = ds["longitude"].values if "longitude" in ds.coords else []
            dlat = abs(float(lat[1] - lat[0])) if len(lat) > 1 else None
            dlon = abs(float(lon[1] - lon[0])) if len(lon) > 1 else None
            rows.append({
                "group": i,
                "sizes": dict(ds.sizes),
                "time_start": t0,
                "time_end": t1,
                "n_time": len(times),
                "dlat": dlat,
                "dlon": dlon,
                "variables": vars_info,
            })
        return rows


def iso_to_datetime(iso: str) -> datetime:
    ts = pd.Timestamp(iso)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime()
