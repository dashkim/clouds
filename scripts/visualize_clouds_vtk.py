#!/usr/bin/env python3
"""VTK viewer for ICON DOM cloud fields (height sweep or time animation)."""
from __future__ import annotations

import argparse
import sys

import netCDF4 as nc
import numpy as np
import vtk

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import add_input_arg, find_icon_dom_nc

EARTH_RADIUS = 1.0


def lonlat_rad_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    coslat = np.cos(lat)
    x = coslat * np.cos(lon)
    y = coslat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z)) * EARTH_RADIUS


def subsample_indices(ncells: int, count: int, seed: int) -> np.ndarray:
    count = min(count, ncells)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ncells, size=count, replace=False))


def build_unstructured_grid(
    lon_bnds: np.ndarray,
    lat_bnds: np.ndarray,
) -> vtk.vtkUnstructuredGrid:
    """lon_bnds, lat_bnds: (ncells, 3) in radians."""
    ncells = lon_bnds.shape[0]
    flat_lon = lon_bnds.reshape(-1)
    flat_lat = lat_bnds.reshape(-1)
    xyz = lonlat_rad_to_xyz(flat_lon, flat_lat)

    points = vtk.vtkPoints()
    for x, y, z in xyz:
        points.InsertNextPoint(float(x), float(y), float(z))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    npts = 3
    for c in range(ncells):
        tri = vtk.vtkTriangle()
        base = c * npts
        for v in range(npts):
            tri.GetPointIds().SetId(v, base + v)
        grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())

    return grid


class CloudViewer:
    def __init__(
        self,
        nc_path: str,
        var_name: str,
        mode: str,
        time_idx: int,
        subsample: int,
        level_stride: int,
        seed: int,
    ) -> None:
        self.ds = nc.Dataset(nc_path, "r")
        self.var = self.ds.variables[var_name]
        self.mode = mode
        self.time_idx = time_idx
        self.level_stride = max(1, level_stride)
        self.cell_idx = subsample_indices(
            self.ds.dimensions["ncells"].size, subsample, seed
        )

        lon_b = self.ds.variables["clon_bnds"][self.cell_idx, :]
        lat_b = self.ds.variables["clat_bnds"][self.cell_idx, :]
        self.grid = build_unstructured_grid(lon_b, lat_b)

        self.scalars = vtk.vtkFloatArray()
        self.scalars.SetName(var_name)
        self.scalars.SetNumberOfTuples(len(self.cell_idx))
        self.grid.GetCellData().SetScalars(self.scalars)

        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.grid)
        self.mapper.SetScalarModeToUseCellData()
        self.mapper.ScalarVisibilityOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.05, 0.05, 0.12)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1280, 720)
        self.render_window.SetWindowName(f"ICON clouds — {var_name}")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        if "height" in self.ds.variables and self.mode == "height":
            self.frames = list(range(0, len(self.ds.variables["height"]), self.level_stride))
            self.frame_label = "height index"
        else:
            self.frames = list(range(len(self.ds.dimensions["time"])))
            self.frame_label = "time index"

        self.frame_pos = 0
        self._update_scalars()
        self.mapper.SetScalarRange(self.scalars.GetRange())

        self.interactor.AddObserver("KeyPressEvent", self._on_key)

    def _read_slice(self, frame: int) -> np.ndarray:
        if self.mode == "height":
            slab = self.var[self.time_idx, frame, :]
        else:
            slab = self.var[frame, 0, :]
        return np.asarray(slab[self.cell_idx], dtype=np.float32)

    def _update_scalars(self) -> None:
        frame = self.frames[self.frame_pos]
        values = self._read_slice(frame)
        for i, v in enumerate(values):
            self.scalars.SetValue(i, float(v))
        self.scalars.Modified()
        self.grid.Modified()
        if self.mode == "height":
            h = self.ds.variables["height"][frame]
            title = f"{self.scalars.GetName()}  height[{frame}]={float(h):.0f} m"
        else:
            title = f"{self.scalars.GetName()}  time[{frame}]"
        self.render_window.SetWindowName(title)

    def _on_key(self, obj, _event) -> None:
        key = obj.GetKeySym()
        if key in ("Right", "n", "N"):
            self.frame_pos = (self.frame_pos + 1) % len(self.frames)
            self._update_scalars()
            self.render_window.Render()
        elif key in ("Left", "p", "P"):
            self.frame_pos = (self.frame_pos - 1) % len(self.frames)
            self._update_scalars()
            self.render_window.Render()

    def _tick(self, *_args) -> None:
        self.frame_pos = (self.frame_pos + 1) % len(self.frames)
        self._update_scalars()
        self.render_window.Render()

    def run(self) -> None:
        self.renderer.ResetCamera()
        self.render_window.Render()
        print("Keys: Right/n = next frame, Left/p = previous, q = quit")
        print(f"Mode: {self.mode}, frames: {len(self.frames)} ({self.frame_label})")
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self._tick)
        self.interactor.CreateRepeatingTimer(400)
        self.interactor.Start()

    def close(self) -> None:
        self.ds.close()


def pick_default_var(ds: nc.Dataset, requested: str | None) -> str:
    if requested:
        if requested not in ds.variables:
            raise KeyError(f"Variable not found: {requested}")
        return requested
    for name in ("clw", "cli", "qr", "hus"):
        if name in ds.variables:
            return name
    raise KeyError("No default cloud variable (clw/cli/qr/hus) in file.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser)
    parser.add_argument("--var", default=None, help="Field name (default: clw)")
    parser.add_argument(
        "--mode",
        choices=("height", "time"),
        default="height",
        help="Animate vertical levels or timesteps",
    )
    parser.add_argument("--time", type=int, default=0, dest="time_idx")
    parser.add_argument("--subsample", type=int, default=50000, help="Number of cells")
    parser.add_argument("--level-stride", type=int, default=5, help="Height level step")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nc_path = find_icon_dom_nc(args.input)
    with nc.Dataset(nc_path, "r") as ds:
        n_time = len(ds.dimensions["time"])
        var_name = pick_default_var(ds, args.var)
        if args.mode == "time" and n_time < 2:
            print("Only one timestep; switching to --mode height.", file=sys.stderr)
            args.mode = "height"

    viewer = CloudViewer(
        str(nc_path),
        var_name,
        args.mode,
        args.time_idx,
        args.subsample,
        args.level_stride,
        args.seed,
    )
    try:
        viewer.run()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
