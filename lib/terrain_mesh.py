"""Terrain mesh generation for the cloud-focused Alps patch."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np
import vtk
from vtk.util import numpy_support

from lib.alps_region import (
    PATCH_CENTER_LAT,
    PATCH_CENTER_LON,
    crop_indices,
    load_crop_multi,
    square_patch_bounds,
)
from lib.meteo_utils import mask_cloud_overlay, pressure_to_height_msl, wind_speed

M_PER_DEG_LAT = 111_320.0


SLAB_MODES = ("clt", "pressure", "wind", "inversion", "inversion_compare", "feature_overlay")


@dataclass
class PatchGrid:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    clt: np.ndarray | None
    center_lon: float
    center_lat: float
    time_values: np.ndarray | None = None
    time_units: str | None = None
    slab_mode: str = "clt"
    overlay: np.ndarray | None = None
    ccb: np.ndarray | None = None
    u_wind: np.ndarray | None = None
    v_wind: np.ndarray | None = None
    inversion: np.ndarray | None = None
    cloud_base_m: np.ndarray | None = None
    cloud_deck_cover: np.ndarray | None = None
    medium_cover: np.ndarray | None = None
    era5_frame_times: list[str] | None = None
    comparison_field: np.ndarray | None = None
    predicted_inversion: np.ndarray | None = None
    feature_t2m: np.ndarray | None = None


def lonlat_to_local_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D lon/lat arrays to local east/north offsets in meters."""
    cos_lat = math.cos(math.radians(center_lat))
    x = (lon - center_lon) * cos_lat * M_PER_DEG_LAT
    y = (lat - center_lat) * M_PER_DEG_LAT
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _fields_for_slab_mode(slab_mode: str, *, load_clt: bool) -> list[str]:
    if slab_mode == "pressure":
        return ["clt", "ccb"]
    if slab_mode == "wind":
        return ["u_10m", "v_10m"]
    if slab_mode == "inversion":
        return ["clt", "ccb"]
    return ["clt"] if load_clt else []


def _build_overlay_series(
    slab_mode: str,
    fields: dict[str, np.ndarray],
    z: np.ndarray,
) -> np.ndarray | None:
    if slab_mode == "clt":
        return fields.get("clt")
    if slab_mode == "pressure":
        clt = fields.get("clt")
        ccb = fields.get("ccb")
        if ccb is None:
            raise ValueError("pressure slab_mode requires ccb in NetCDF.")
        n_time = ccb.shape[0]
        overlay = np.empty((n_time, z.shape[0], z.shape[1]), dtype=np.float32)
        for t in range(n_time):
            height = pressure_to_height_msl(ccb[t])
            if clt is not None:
                height = mask_cloud_overlay(height, clt[t])
            overlay[t] = height
        return overlay
    if slab_mode == "wind":
        u = fields.get("u_10m")
        v = fields.get("v_10m")
        if u is None or v is None:
            raise ValueError("wind slab_mode requires u_10m and v_10m in NetCDF.")
        n_time = u.shape[0]
        overlay = np.empty((n_time, z.shape[0], z.shape[1]), dtype=np.float32)
        for t in range(n_time):
            overlay[t] = wind_speed(u[t], v[t])
        return overlay
    if slab_mode == "inversion":
        clt = fields.get("clt")
        if clt is None:
            raise ValueError("inversion slab_mode requires pre-built PatchGrid from patch_fields.")
        return None
    raise ValueError(f"Unknown slab_mode: {slab_mode}")


def load_patch_grid(
    nc_path: Path,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    center_lon: float = PATCH_CENTER_LON,
    center_lat: float = PATCH_CENTER_LAT,
    load_clt: bool = True,
    slab_mode: str = "clt",
    stride: int = 1,
) -> PatchGrid:
    if slab_mode not in SLAB_MODES:
        raise ValueError(f"slab_mode must be one of {SLAB_MODES}")

    with nc.Dataset(nc_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            lon_min, lon_max, lat_min, lat_max,
        )

    var_names = _fields_for_slab_mode(slab_mode, load_clt=load_clt)
    fields, meta = load_crop_multi(nc_path, lon_sl, lat_sl, var_names, stride=stride)
    overlay = _build_overlay_series(slab_mode, fields, meta.z)
    return PatchGrid(
        lon=meta.lon,
        lat=meta.lat,
        z=meta.z,
        clt=fields.get("clt"),
        center_lon=center_lon,
        center_lat=center_lat,
        time_values=meta.time_values,
        time_units=meta.time_units,
        slab_mode=slab_mode,
        overlay=overlay,
        ccb=fields.get("ccb"),
        u_wind=fields.get("u_10m"),
        v_wind=fields.get("v_10m"),
    )


def _build_structured_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vert_exag: float,
    *,
    z_offset: float = 0.0,
    point_scalars: np.ndarray | None = None,
    scalar_name: str = "elevation",
) -> vtk.vtkStructuredGrid:
    n_lat, n_lon = z.shape
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(n_lon, n_lat, 1)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_lon * n_lat)
    scalars = np.empty(n_lon * n_lat, dtype=np.float32)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            z_val = z[i, j]
            if np.isfinite(z_val):
                z_plot = z_val * vert_exag + z_offset
            else:
                z_plot = 0.0
            pts.SetPoint(idx, float(x[j]), float(y[i]), float(z_plot))
            if point_scalars is not None:
                scalars[idx] = point_scalars[i, j]
            else:
                scalars[idx] = z_val if np.isfinite(z_val) else np.nan
            idx += 1

    grid.SetPoints(pts)
    arr = numpy_support.numpy_to_vtk(scalars, deep=True)
    arr.SetName(scalar_name)
    grid.GetPointData().SetScalars(arr)
    return grid


def structured_grid_to_surface(grid: vtk.vtkStructuredGrid) -> vtk.vtkPolyData:
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(grid)
    geom.Update()
    return geom.GetOutput()


def build_terrain_surface(grid: PatchGrid, vert_exag: float) -> vtk.vtkPolyData:
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    structured = _build_structured_grid(x, y, grid.z, vert_exag, scalar_name="elevation")
    return structured_grid_to_surface(structured)


def build_scalar_slab(
    grid: PatchGrid,
    values_2d: np.ndarray,
    *,
    vert_exag: float,
    slab_height: float,
    scalar_name: str = "scalar",
) -> vtk.vtkPolyData:
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    z_offset = slab_height * vert_exag
    structured = _build_structured_grid(
        x, y, grid.z, vert_exag,
        z_offset=z_offset,
        point_scalars=values_2d.astype(np.float32),
        scalar_name=scalar_name,
    )
    return structured_grid_to_surface(structured)


def build_cloud_slab(
    grid: PatchGrid,
    clt_2d: np.ndarray,
    *,
    vert_exag: float,
    slab_height: float,
) -> vtk.vtkPolyData:
    return build_scalar_slab(
        grid,
        clt_2d,
        vert_exag=vert_exag,
        slab_height=slab_height,
        scalar_name="clt",
    )


def _deck_height_m(
    grid: PatchGrid,
    cloud_base_2d: np.ndarray | None,
    *,
    fallback_percentile: float = 30.0,
) -> np.ndarray:
    """Per-cell cloud deck height MSL; fallback to valley elevation percentile."""
    z = grid.z
    if cloud_base_2d is not None and np.any(np.isfinite(cloud_base_2d)):
        deck = np.asarray(cloud_base_2d, dtype=np.float32).copy()
        valid_cbh = np.isfinite(deck) & (deck > 0.0)
        if np.any(valid_cbh):
            fallback = float(np.nanpercentile(z[np.isfinite(z)], fallback_percentile))
            deck[~valid_cbh] = fallback
            return deck
    return np.full(z.shape, float(np.nanpercentile(z[np.isfinite(z)], fallback_percentile)), dtype=np.float32)


def _fog_deck_height_m(
    terrain_z: np.ndarray,
    cloud_base_2d: np.ndarray,
    cover_2d: np.ndarray,
    *,
    cover_min: float,
    uniform: bool,
) -> np.ndarray:
    """Cloud deck elevation MSL (ERA5 cbh is above ground, not MSL)."""
    from lib.inversion import cbh_agl_to_msl, inversion_fog_deck_msl

    cbh = np.asarray(cloud_base_2d, dtype=np.float32)
    cover = np.asarray(cover_2d, dtype=np.float32)
    z = np.asarray(terrain_z, dtype=np.float32)
    valid_cbh = np.isfinite(cbh) & (cbh > 0.0)
    if not np.any(valid_cbh):
        return np.zeros_like(cbh)

    if uniform:
        deck = inversion_fog_deck_msl(z, cbh, cover, cover_min=cover_min)
        return np.full(cbh.shape, deck, dtype=np.float32)

    return cbh_agl_to_msl(cbh, z)


def build_inversion_cloud_layer(
    grid: PatchGrid,
    cover_2d: np.ndarray,
    cloud_base_2d: np.ndarray,
    *,
    vert_exag: float,
    cover_min: float = 20.0,
    uniform_deck: bool = True,
    scalar_name: str = "cloud_deck",
) -> vtk.vtkPolyData:
    """
    Horizontal cloud layer at inversion height (MSL).

    ERA5 cbh is height above ground; deck is placed so terrain intersects
    the fog bank and peaks rise above it.
    """
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    cover = np.asarray(cover_2d, dtype=np.float32).copy()
    deck_z = _fog_deck_height_m(
        grid.z,
        cloud_base_2d,
        cover,
        cover_min=cover_min,
        uniform=uniform_deck,
    )

    hide = ~np.isfinite(grid.z) | ~np.isfinite(cover) | (cover < cover_min)
    cover[hide] = -1.0

    n_lat, n_lon = grid.z.shape
    vtk_grid = vtk.vtkStructuredGrid()
    vtk_grid.SetDimensions(n_lon, n_lat, 1)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_lon * n_lat)
    scalars = np.empty(n_lon * n_lat, dtype=np.float32)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            z_plot = float(deck_z[i, j] * vert_exag)
            pts.SetPoint(idx, float(x[j]), float(y[i]), z_plot)
            scalars[idx] = cover[i, j]
            idx += 1

    vtk_grid.SetPoints(pts)
    arr = numpy_support.numpy_to_vtk(scalars, deep=True)
    arr.SetName(scalar_name)
    vtk_grid.GetPointData().SetScalars(arr)
    return structured_grid_to_surface(vtk_grid)


def build_uniform_deck_surface(
    grid: PatchGrid,
    deck_msl: float,
    cover_2d: np.ndarray,
    *,
    vert_exag: float,
    cover_min: float = 20.0,
    peak_margin_m: float = 80.0,
    scalar_name: str = "deck",
) -> vtk.vtkPolyData:
    """Horizontal deck at a fixed MSL height; peaks above deck are masked out."""
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    cover = np.asarray(cover_2d, dtype=np.float32)
    z = grid.z
    z_plot = float(deck_msl) * vert_exag

    hide = (
        ~np.isfinite(z)
        | ~np.isfinite(cover)
        | (cover < cover_min)
        | (z > float(deck_msl) + peak_margin_m)
    )

    n_lat, n_lon = z.shape
    vtk_grid = vtk.vtkStructuredGrid()
    vtk_grid.SetDimensions(n_lon, n_lat, 1)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_lon * n_lat)
    scalars = np.empty(n_lon * n_lat, dtype=np.float32)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            pts.SetPoint(idx, float(x[j]), float(y[i]), z_plot)
            scalars[idx] = 0.0 if hide[i, j] else 1.0
            idx += 1

    vtk_grid.SetPoints(pts)
    arr = numpy_support.numpy_to_vtk(scalars, deep=True)
    arr.SetName(scalar_name)
    vtk_grid.GetPointData().SetScalars(arr)
    return structured_grid_to_surface(vtk_grid)


def build_deck_gap_layer(
    grid: PatchGrid,
    actual_msl: float,
    predicted_msl: float,
    agreement_2d: np.ndarray,
    *,
    vert_exag: float,
    min_gap_m: float = 8.0,
    scalar_name: str = "gap_class",
) -> vtk.vtkPolyData | None:
    """
    Vertical slab between actual and predicted deck heights.

    Scalar classes: 1=TP (green), 2=FP (red), 3=FN (blue), 0=hidden.
    """
    gap_m = abs(float(actual_msl) - float(predicted_msl))
    agree = np.asarray(agreement_2d, dtype=np.float32)
    has_class = np.any(np.isin(agree, (1.0, 2.0, 3.0)) & np.isfinite(agree))

    if gap_m < min_gap_m and not has_class:
        return None

    mid_msl = 0.5 * (float(actual_msl) + float(predicted_msl))
    half_m = max(gap_m * 0.5, 18.0) if gap_m >= min_gap_m else 18.0
    z_bot = (mid_msl - half_m) * vert_exag
    z_top = (mid_msl + half_m) * vert_exag

    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    n_lat, n_lon = grid.z.shape
    n_layers = 2
    vtk_grid = vtk.vtkStructuredGrid()
    vtk_grid.SetDimensions(n_lon, n_lat, n_layers)

    pts = vtk.vtkPoints()
    n_pts = n_lon * n_lat * n_layers
    pts.SetNumberOfPoints(n_pts)
    scalars = np.zeros(n_pts, dtype=np.float32)

    idx = 0
    for k, z_plot in enumerate((z_bot, z_top)):
        for i in range(n_lat):
            for j in range(n_lon):
                cls = float(agree[i, j]) if np.isfinite(agree[i, j]) else 0.0
                if cls not in (1.0, 2.0, 3.0):
                    cls = 0.0
                pts.SetPoint(idx, float(x[j]), float(y[i]), float(z_plot))
                scalars[idx] = cls
                idx += 1

    vtk_grid.SetPoints(pts)
    arr = numpy_support.numpy_to_vtk(scalars, deep=True)
    arr.SetName(scalar_name)
    vtk_grid.GetPointData().SetScalars(arr)
    return structured_grid_to_surface(vtk_grid)


def build_horizontal_level_plane(
    grid: PatchGrid,
    deck_msl: float,
    *,
    vert_exag: float,
    scalar_name: str = "level",
) -> vtk.vtkPolyData:
    """Flat horizontal reference plane at a single MSL height (m)."""
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    z_plot = float(deck_msl) * vert_exag
    n_lat, n_lon = grid.z.shape

    vtk_grid = vtk.vtkStructuredGrid()
    vtk_grid.SetDimensions(n_lon, n_lat, 1)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_lon * n_lat)
    scalars = np.ones(n_lon * n_lat, dtype=np.float32)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            pts.SetPoint(idx, float(x[j]), float(y[i]), z_plot)
            idx += 1

    vtk_grid.SetPoints(pts)
    arr = numpy_support.numpy_to_vtk(scalars, deep=True)
    arr.SetName(scalar_name)
    vtk_grid.GetPointData().SetScalars(arr)
    return structured_grid_to_surface(vtk_grid)


def build_valley_fog_slab(
    grid: PatchGrid,
    cover_2d: np.ndarray,
    cloud_base_2d: np.ndarray | None,
    *,
    vert_exag: float,
    slab_height: float = 400.0,
    margin_m: float = 80.0,
    cover_min: float = 15.0,
    scalar_name: str = "cloud_deck",
) -> vtk.vtkPolyData:
    """Terrain-following fog (legacy); prefer build_inversion_cloud_layer."""
    cover = np.asarray(cover_2d, dtype=np.float32).copy()
    terrain = grid.z
    valid = np.isfinite(terrain)

    if cloud_base_2d is not None:
        cbh = np.asarray(cloud_base_2d, dtype=np.float32)
        above_deck = valid & np.isfinite(cbh) & (cbh > 0) & (terrain > cbh + margin_m)
        cover[above_deck] = np.nan

    hide = ~valid | (cover < cover_min)
    if cloud_base_2d is not None:
        hide = hide | above_deck
    cover[hide] = -1.0

    return build_scalar_slab(
        grid,
        cover,
        vert_exag=vert_exag,
        slab_height=slab_height,
        scalar_name=scalar_name,
    )


def build_cloud_deck_slab(
    grid: PatchGrid,
    cover_2d: np.ndarray,
    cloud_base_2d: np.ndarray | None,
    *,
    vert_exag: float,
    margin_m: float = 50.0,
    slab_height: float = 400.0,
    scalar_name: str = "cloud_deck",
) -> vtk.vtkPolyData:
    """Horizontal inversion cloud layer when cbh is available."""
    if cloud_base_2d is not None and np.any(np.isfinite(cloud_base_2d)):
        return build_inversion_cloud_layer(
            grid,
            cover_2d,
            cloud_base_2d,
            vert_exag=vert_exag,
            scalar_name=scalar_name,
        )
    return build_valley_fog_slab(
        grid,
        cover_2d,
        cloud_base_2d,
        vert_exag=vert_exag,
        slab_height=slab_height,
        margin_m=margin_m,
        scalar_name=scalar_name,
    )


def build_inversion_terrain_overlay(
    grid: PatchGrid,
    inversion_2d: np.ndarray,
    *,
    vert_exag: float,
    scalar_name: str = "inversion",
) -> vtk.vtkPolyData:
    """Terrain surface colored by inversion mask (ridge highlight)."""
    return build_feature_terrain_surface(
        grid,
        inversion_2d,
        vert_exag=vert_exag,
        scalar_name=scalar_name,
    )


def build_feature_terrain_surface(
    grid: PatchGrid,
    values_2d: np.ndarray,
    *,
    vert_exag: float,
    scalar_name: str = "feature",
) -> vtk.vtkPolyData:
    """Terrain surface colored by a 2D scalar field (e.g. LCC, T2m, wind speed)."""
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    values = np.asarray(values_2d, dtype=np.float32)
    values[~np.isfinite(grid.z)] = np.nan
    structured = _build_structured_grid(
        x, y, grid.z, vert_exag,
        point_scalars=values,
        scalar_name=scalar_name,
    )
    return structured_grid_to_surface(structured)


def patch_centroid_metric(
    grid: PatchGrid,
    vert_exag: float,
) -> tuple[float, float, float]:
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    valid = np.isfinite(grid.z)
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    x2d, y2d = np.meshgrid(x, y)
    cx = float(x2d[valid].mean())
    cy = float(y2d[valid].mean())
    cz = float((grid.z[valid] * vert_exag).mean())
    return cx, cy, cz


def patch_horizontal_span_m(grid: PatchGrid) -> float:
    x, y = lonlat_to_local_meters(grid.lon, grid.lat, grid.center_lon, grid.center_lat)
    return float(max(x.max() - x.min(), y.max() - y.min()))


def write_vtp(polydata: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()
