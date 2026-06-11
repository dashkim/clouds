"""Unified patch field loading: ICON terrain + cloud labels (ICON or ERA5)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lib.alps_region import crop_indices, default_patch_stride, resolve_patch_bounds
from lib.era5_io import (
    cbh_at_index,
    era5_to_icon_grid,
    iso_to_datetime,
    load_era5_hourly_fields,
    nearest_era5_index,
    step_dataset,
)
from lib.inversion import InversionParams, cloud_base_from_ccb, detect_inversion_mask
from lib.netcdf import fill_value, mask_invalid
from lib.paths import find_era5_grib, find_lonlat_nc
from lib.terrain_mesh import PatchGrid

import netCDF4 as nc


@dataclass
class PatchFields:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    clt: np.ndarray | None
    ccb: np.ndarray | None
    inversion: np.ndarray
    cloud_base_m: np.ndarray | None
    low_cover: np.ndarray | None
    medium_cover: np.ndarray | None
    center_lon: float
    center_lat: float
    time_values: np.ndarray | None
    time_units: str | None
    label_source: str
    era5_times: np.ndarray | None = None
    era5_frame_times: list[str] | None = None

    @property
    def n_time(self) -> int:
        return self.inversion.shape[0]


def _load_icon_terrain(
    nc_path: Path,
    lon_sl: slice,
    lat_sl: slice,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = max(1, stride)
    lon_sl_s = slice(lon_sl.start, lon_sl.stop, st)
    lat_sl_s = slice(lat_sl.start, lat_sl.stop, st)
    with nc.Dataset(nc_path, "r") as ds:
        lon = np.asarray(ds.variables["lon"][lon_sl_s], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][lat_sl_s], dtype=np.float64)
        z_var = ds.variables["z_ifc"]
        z = mask_invalid(
            np.asarray(z_var[lat_sl_s, lon_sl_s], dtype=np.float32),
            fill_value(z_var),
        )
    return lon, lat, z


def _load_icon_series(
    nc_path: Path,
    lon_sl: slice,
    lat_sl: slice,
    var_name: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None, str | None, int]:
    st = max(1, stride)
    lon_sl_s = slice(lon_sl.start, lon_sl.stop, st)
    lat_sl_s = slice(lat_sl.start, lat_sl.stop, st)
    with nc.Dataset(nc_path, "r") as ds:
        n_time = len(ds.dimensions["time"])
        var = ds.variables[var_name]
        fill = fill_value(var)
        n_lat = len(range(*lat_sl_s.indices(len(ds.variables["lat"]))))
        n_lon = len(range(*(lon_sl_s.indices(len(ds.variables["lon"])))))
        arr = np.empty((n_time, n_lat, n_lon), dtype=np.float32)
        for t in range(n_time):
            slab = var[t]
            while slab.ndim > 2:
                slab = slab[0]
            arr[t] = mask_invalid(np.asarray(slab[lat_sl_s, lon_sl_s], dtype=np.float32), fill)
        time_var = ds.variables.get("time")
        time_values = np.asarray(time_var[:], dtype=np.float64) if time_var is not None else None
        time_units = getattr(time_var, "units", None) if time_var is not None else None
    return arr, time_values, time_units, n_time


def _build_inversion_series(
    z: np.ndarray,
    *,
    cloud_cover: np.ndarray,
    low_cover: np.ndarray | None,
    medium_cover: np.ndarray | None,
    cloud_base_m: np.ndarray | None,
    z_min_m: float,
    inversion_params: InversionParams | None = None,
) -> np.ndarray:
    params = inversion_params or InversionParams(z_min_m=z_min_m)
    n_time = cloud_cover.shape[0]
    inv = np.empty((n_time, z.shape[0], z.shape[1]), dtype=np.float32)
    base_series = cloud_base_m
    for t in range(n_time):
        base_t = None if base_series is None else base_series[t] if base_series.ndim == 3 else base_series
        inv[t] = detect_inversion_mask(
            z,
            cloud_cover=cloud_cover[t],
            low_cover=None if low_cover is None else low_cover[t],
            medium_cover=None if medium_cover is None else medium_cover[t],
            cloud_base_m=base_t,
            params=params,
        )
    return inv


def load_patch_fields(
    *,
    region: str = "east_core",
    label_source: str = "era5",
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    z_min_m: float = 1000.0,
    era5_time: datetime | str | None = None,
    era5_start: datetime | str | None = None,
    era5_n_frames: int | None = None,
    max_time_steps: int | None = None,
) -> PatchFields:
    """Load terrain from ICON; cloud labels and inversion from ERA5 or ICON."""
    icon_path = icon_nc or find_lonlat_nc()
    era5_path = era5_grib or find_era5_grib()
    stride = stride if stride is not None else default_patch_stride(region)

    lon_min, lon_max, lat_min, lat_max, center_lon, center_lat = resolve_patch_bounds(region=region)

    with nc.Dataset(icon_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            lon_min, lon_max, lat_min, lat_max,
        )

    lon, lat, z = _load_icon_terrain(icon_path, lon_sl, lat_sl, stride)
    era5_frame_times: list[str] | None = None

    if label_source == "icon":
        clt, time_values, time_units, _ = _load_icon_series(icon_path, lon_sl, lat_sl, "clt", stride)
        try:
            ccb, _, _, _ = _load_icon_series(icon_path, lon_sl, lat_sl, "ccb", stride)
        except KeyError:
            ccb = None
        low = clt
        med = None
        base_series = None
        if ccb is not None:
            base_series = np.empty_like(ccb)
            for t in range(ccb.shape[0]):
                base_series[t] = cloud_base_from_ccb(ccb[t])
        era5_times = None
    elif label_source == "era5":
        ds_hourly = load_era5_hourly_fields(
            era5_path,
            lon_min=lon_min - 0.5,
            lon_max=lon_max + 0.5,
            lat_min=lat_min - 0.5,
            lat_max=lat_max + 0.5,
            variables=("tcc", "lcc", "mcc"),
        )
        era5_times = pd.to_datetime(ds_hourly["time"].values).to_numpy()

        if era5_time is not None:
            target = iso_to_datetime(era5_time) if isinstance(era5_time, str) else era5_time
            indices = [nearest_era5_index(era5_times, target)]
            era5_frame_times = [str(pd.Timestamp(era5_times[i])) for i in indices]
            time_values = None
            time_units = None
        elif era5_start is not None:
            start = iso_to_datetime(era5_start) if isinstance(era5_start, str) else era5_start
            start_idx = nearest_era5_index(era5_times, start)
            n_frames = era5_n_frames or 8
            indices = list(range(start_idx, min(start_idx + n_frames, len(era5_times))))
            era5_frame_times = [str(pd.Timestamp(era5_times[i])) for i in indices]
            time_values = None
            time_units = "ERA5 hourly UTC"
        else:
            indices = list(range(len(era5_times)))
            if max_time_steps is not None:
                indices = indices[:max_time_steps]
            time_values = None
            time_units = "ERA5 hourly UTC"

        n_time = len(indices)
        clt = np.empty((n_time, lat.size, lon.size), dtype=np.float32)
        low = np.empty_like(clt)
        med = np.empty_like(clt)
        base_series = np.empty_like(clt)
        ccb = None

        era5_lat = np.asarray(ds_hourly["latitude"].values, dtype=np.float64)
        era5_lon = np.asarray(ds_hourly["longitude"].values, dtype=np.float64)
        tcc_all = np.asarray(ds_hourly["tcc"].values, dtype=np.float32)
        lcc_all = np.asarray(ds_hourly["lcc"].values, dtype=np.float32)
        mcc_all = np.asarray(ds_hourly["mcc"].values, dtype=np.float32)
        ds_step = step_dataset(era5_path)
        cbh_lat = np.asarray(ds_step["latitude"].values, dtype=np.float64)
        cbh_lon = np.asarray(ds_step["longitude"].values, dtype=np.float64)

        for out_t, era5_t in enumerate(indices):
            clt[out_t] = era5_to_icon_grid(tcc_all[era5_t], era5_lat, era5_lon, lat, lon)
            low[out_t] = era5_to_icon_grid(lcc_all[era5_t], era5_lat, era5_lon, lat, lon)
            med[out_t] = era5_to_icon_grid(mcc_all[era5_t], era5_lat, era5_lon, lat, lon)
            cbh_field = cbh_at_index(era5_path, era5_t, era5_times)
            base_series[out_t] = era5_to_icon_grid(cbh_field, cbh_lat, cbh_lon, lat, lon)
    else:
        raise ValueError(f"label_source must be 'icon' or 'era5', got {label_source!r}")

    inversion = _build_inversion_series(
        z,
        cloud_cover=clt,
        low_cover=low,
        medium_cover=med,
        cloud_base_m=base_series,
        z_min_m=z_min_m,
    )

    return PatchFields(
        lon=lon,
        lat=lat,
        z=z,
        clt=clt,
        ccb=ccb,
        inversion=inversion,
        cloud_base_m=base_series,
        low_cover=low,
        medium_cover=med,
        center_lon=center_lon,
        center_lat=center_lat,
        time_values=time_values,
        time_units=time_units,
        label_source=label_source,
        era5_times=era5_times,
        era5_frame_times=era5_frame_times,
    )


def era5_to_icon_grid_cbh(
    ds_step,
    target_ts: pd.Timestamp,
    icon_lat: np.ndarray,
    icon_lon: np.ndarray,
) -> np.ndarray:
    from lib.era5_io import era5_to_icon_grid

    field = cbh_at_time(ds_step, target_ts)
    era5_lat = np.asarray(ds_step["latitude"].values, dtype=np.float64)
    era5_lon = np.asarray(ds_step["longitude"].values, dtype=np.float64)
    return era5_to_icon_grid(field, era5_lat, era5_lon, icon_lat, icon_lon)


def patch_fields_to_grid(fields: PatchFields, *, slab_mode: str = "inversion") -> PatchGrid:
    """Convert PatchFields to PatchGrid for VTK rendering."""
    overlay = fields.inversion if slab_mode == "inversion" else fields.clt
    if overlay is None:
        raise ValueError(f"No overlay for slab_mode={slab_mode}")

    deck_cover = fields.low_cover if fields.low_cover is not None else fields.clt

    return PatchGrid(
        lon=fields.lon,
        lat=fields.lat,
        z=fields.z,
        clt=fields.clt,
        center_lon=fields.center_lon,
        center_lat=fields.center_lat,
        time_values=fields.time_values,
        time_units=fields.time_units,
        slab_mode=slab_mode,
        overlay=overlay,
        ccb=fields.ccb,
        inversion=fields.inversion,
        cloud_base_m=fields.cloud_base_m,
        cloud_deck_cover=deck_cover,
        medium_cover=fields.medium_cover,
        era5_frame_times=fields.era5_frame_times,
    )


def load_patch_grid_inversion(
    *,
    region: str = "east_core",
    label_source: str = "era5",
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    z_min_m: float = 1000.0,
    era5_time: datetime | str | None = None,
    era5_start: datetime | str | None = None,
    era5_n_frames: int | None = None,
    max_time_steps: int | None = None,
) -> PatchGrid:
    fields = load_patch_fields(
        region=region,
        label_source=label_source,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        stride=stride,
        z_min_m=z_min_m,
        era5_time=era5_time,
        era5_start=era5_start,
        era5_n_frames=era5_n_frames,
        max_time_steps=max_time_steps,
    )
    return patch_fields_to_grid(fields, slab_mode="inversion")
