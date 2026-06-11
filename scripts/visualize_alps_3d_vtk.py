#!/usr/bin/env python3
"""Interactive VTK preview: 3D Bavarian Alps terrain + pseudo-3D cloud slab."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import (
    DEFAULT_ELEV_MIN,
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    crop_indices,
    load_crop_multi,
)
from lib.netcdf import wall_clock
from lib.paths import CLOUDS_ROOT, add_input_arg, find_lonlat_nc
from lib.vtk_alps_scene import Alps3DScene, AlpsSceneConfig


class Alps3DViewer:
    def __init__(
        self,
        scene: Alps3DScene,
        time_var,
        time_values,
        time_units: str | None,
        interval_ms: int,
        start: int,
        screenshot_dir: Path,
    ) -> None:
        self.scene = scene
        self.time_var = time_var
        self.time_values = time_values
        self.time_units = time_units
        self.interval_ms = interval_ms
        self.playing = True
        self.frame_pos = max(0, min(start, scene.clt_series.shape[0] - 1))
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        scene.configure_render_window(1280, 720, offscreen=False)
        scene.set_camera_orbit(self.frame_pos, scene.clt_series.shape[0])
        scene.update_frame(self.frame_pos)
        scene.render()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(scene.render_window)
        self.interactor.AddObserver("KeyPressEvent", self._on_key)

    def _title(self) -> str:
        ts = wall_clock(
            self.time_var,
            self.frame_pos,
            time_values=self.time_values,
            time_units=self.time_units,
        ) or f"timestep {self.frame_pos}"
        n = self.scene.clt_series.shape[0]
        status = "" if self.playing else " (paused)"
        return f"Alps 3D — clt  {ts}  [{self.frame_pos + 1}/{n}]{status}"

    def _step(self, delta: int) -> None:
        n = self.scene.clt_series.shape[0]
        self.frame_pos = (self.frame_pos + delta) % n
        self.scene.update_frame(self.frame_pos)
        self.scene.set_camera_orbit(self.frame_pos, n)
        self.scene.render()
        self.scene.render_window.SetWindowName(self._title())

    def _on_key(self, obj, _event) -> None:
        key = obj.GetKeySym()
        if key in ("Right", "n", "N"):
            self._step(1)
        elif key in ("Left", "p", "P"):
            self._step(-1)
        elif key == "space":
            self.playing = not self.playing
            self.scene.render_window.SetWindowName(self._title())
            self.scene.render()
        elif key in ("s", "S"):
            path = self.screenshot_dir / f"alps_3d_preview_{self.frame_pos:04d}.png"
            self.scene.screenshot_png(str(path))
            print(f"Saved {path}")
        elif key in ("q", "Q"):
            self.interactor.TerminateApp()

    def _tick(self, *_args) -> None:
        if self.playing:
            self._step(1)

    def run(self) -> None:
        self.scene.render_window.SetWindowName(self._title())
        print("Keys: Right/n = next, Left/p = previous, Space = pause, s = screenshot, q = quit")
        print(f"Frames: {self.scene.clt_series.shape[0]} timesteps, interval {self.interval_ms} ms")
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self._tick)
        self.interactor.CreateRepeatingTimer(self.interval_ms)
        self.interactor.Start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--elev-min", type=float, default=DEFAULT_ELEV_MIN)
    parser.add_argument("--vert-exag", type=float, default=3.5)
    parser.add_argument("--slab-height", type=float, default=600.0)
    parser.add_argument("--stride", type=int, default=1, help="Lon/lat subsample step")
    parser.add_argument("--start", type=int, default=0, help="Initial timestep index")
    parser.add_argument("--interval-ms", type=int, default=400)
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=CLOUDS_ROOT / "output" / "frames",
    )
    args = parser.parse_args()

    nc_path = find_lonlat_nc(args.input)
    with nc.Dataset(nc_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            args.lon_min,
            args.lon_max,
            args.lat_min,
            args.lat_max,
        )

    fields, meta = load_crop_multi(
        nc_path, lon_sl, lat_sl, ["clt"], stride=args.stride,
    )
    config = AlpsSceneConfig(
        elev_min=args.elev_min,
        vert_exag=args.vert_exag,
        slab_height=args.slab_height,
    )
    scene = Alps3DScene(meta.lon, meta.lat, meta.z, fields["clt"], config)
    viewer = Alps3DViewer(
        scene,
        meta.time_var,
        meta.time_values,
        meta.time_units,
        args.interval_ms,
        args.start,
        args.screenshot_dir,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
