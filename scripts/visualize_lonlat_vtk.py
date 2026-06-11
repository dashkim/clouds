#!/usr/bin/env python3
"""VTK viewer for 2D lon/lat fields — time animation movie."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import vtk
from vtk.util import numpy_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.netcdf import fill_value, finite_range, mask_invalid, wall_clock
from lib.paths import add_input_arg, find_lonlat_nc


def build_image_grid(lon: np.ndarray, lat: np.ndarray) -> vtk.vtkImageData:
    n_lon = len(lon)
    n_lat = len(lat)
    image = vtk.vtkImageData()
    image.SetDimensions(n_lon, n_lat, 1)
    image.SetOrigin(float(lon[0]), float(lat[0]), 0.0)
    if n_lon > 1:
        x_spacing = float(lon[-1] - lon[0]) / (n_lon - 1)
    else:
        x_spacing = 1.0
    if n_lat > 1:
        y_spacing = float(lat[-1] - lat[0]) / (n_lat - 1)
    else:
        y_spacing = 1.0
    image.SetSpacing(x_spacing, y_spacing, 1.0)
    return image


def pick_default_var(ds: nc.Dataset, requested: str | None) -> str:
    if requested:
        if requested not in ds.variables:
            raise KeyError(f"Variable not found: {requested}")
        return requested
    for name in ("clt", "ccb", "t_ctop", "u_10m", "cct", "t_cbase", "runoff_s"):
        if name in ds.variables:
            return name
    raise KeyError("No default 2D field found in file.")


class LonLatViewer:
    def __init__(
        self,
        nc_path: str,
        var_name: str,
        stride: int,
        start: int,
        interval_ms: int,
    ) -> None:
        self.ds = nc.Dataset(nc_path, "r")
        self.var = self.ds.variables[var_name]
        self.var_name = var_name
        self.stride = max(1, stride)
        self.interval_ms = interval_ms
        self.playing = True

        lon = self.ds.variables["lon"][:: self.stride]
        lat = self.ds.variables["lat"][:: self.stride]
        self.lon = np.asarray(lon, dtype=np.float64)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.fill = fill_value(self.var)

        self.time_var = self.ds.variables.get("time")

        self.frames = list(range(len(self.ds.dimensions["time"])))
        self.frame_pos = max(0, min(start, len(self.frames) - 1))

        self.image = build_image_grid(self.lon, self.lat)
        self.scalars = vtk.vtkFloatArray()
        self.scalars.SetName(var_name)
        self.image.GetPointData().SetScalars(self.scalars)

        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(self.image)
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.ScalarVisibilityOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.05, 0.05, 0.12)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1280, 720)
        self.render_window.SetWindowName(f"LON/LAT — {var_name}")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.AddObserver("KeyPressEvent", self._on_key)

        self._update_scalars()
        vmin, vmax = finite_range(self._read_slice(self.frames[self.frame_pos]))
        self.mapper.SetScalarRange(vmin, vmax)

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.mapper.GetLookupTable())
        self.scalar_bar.SetTitle(var_name)
        self.scalar_bar.SetNumberOfLabels(5)
        self.renderer.AddActor(self.scalar_bar)

    def _read_slice(self, time_idx: int) -> np.ndarray:
        slab = self.var[time_idx]
        while slab.ndim > 2:
            slab = slab[0]
        slab = np.asarray(slab[:: self.stride, :: self.stride], dtype=np.float32)
        return mask_invalid(slab, self.fill)

    def _wall_clock(self, time_idx: int) -> str:
        label = wall_clock(self.time_var, time_idx)
        return label if label else f"time[{time_idx}]"

    def _update_scalars(self) -> None:
        time_idx = self.frames[self.frame_pos]
        values = self._read_slice(time_idx)
        vtk_arr = numpy_support.numpy_to_vtk(
            values.ravel(order="C"),
            array_type=numpy_support.get_vtk_array_type(values.dtype),
        )
        vtk_arr.SetName(self.var_name)
        self.image.GetPointData().SetScalars(vtk_arr)
        self.scalars = vtk_arr
        self.image.Modified()

        n = len(self.frames)
        title = (
            f"{self.var_name}  {self._wall_clock(time_idx)}  "
            f"[{self.frame_pos + 1}/{n}]"
        )
        if not self.playing:
            title += "  (paused)"
        self.render_window.SetWindowName(title)

    def _step(self, delta: int) -> None:
        self.frame_pos = (self.frame_pos + delta) % len(self.frames)
        self._update_scalars()
        self.render_window.Render()

    def _on_key(self, obj, _event) -> None:
        key = obj.GetKeySym()
        if key in ("Right", "n", "N"):
            self._step(1)
        elif key in ("Left", "p", "P"):
            self._step(-1)
        elif key == "space":
            self.playing = not self.playing
            self._update_scalars()
            self.render_window.Render()
        elif key in ("q", "Q"):
            self.interactor.TerminateApp()

    def _tick(self, *_args) -> None:
        if self.playing:
            self._step(1)

    def run(self) -> None:
        self.renderer.ResetCamera()
        self.render_window.Render()
        print("Keys: Right/n = next, Left/p = previous, Space = pause/resume, q = quit")
        print(f"Frames: {len(self.frames)} timesteps, interval {self.interval_ms} ms")
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self._tick)
        self.interactor.CreateRepeatingTimer(self.interval_ms)
        self.interactor.Start()

    def close(self) -> None:
        self.ds.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    parser.add_argument("--var", default=None, help="Field name (default: clt)")
    parser.add_argument("--stride", type=int, default=1, help="Lon/lat subsample step")
    parser.add_argument("--start", type=int, default=0, help="Initial timestep index")
    parser.add_argument("--interval-ms", type=int, default=400, help="Playback interval (ms)")
    args = parser.parse_args()

    nc_path = find_lonlat_nc(args.input)
    with nc.Dataset(nc_path, "r") as ds:
        var_name = pick_default_var(ds, args.var)

    viewer = LonLatViewer(
        str(nc_path),
        var_name,
        args.stride,
        args.start,
        args.interval_ms,
    )
    try:
        viewer.run()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
