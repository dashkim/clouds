#!/usr/bin/env python3
"""Interactive VTK preview: patch terrain mesh + pseudo-3D cloud slab."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import (
    PATCH_CENTER_LAT,
    PATCH_CENTER_LON,
    PATCH_HALF_SIZE,
    default_patch_vert_exag,
    resolve_patch_bounds,
)
from lib.netcdf import wall_clock
from lib.patch_fields import load_patch_grid_inversion
from lib.paths import CLOUDS_ROOT, add_era5_arg, add_input_arg, find_lonlat_nc
from lib.terrain_mesh import SLAB_MODES, load_patch_grid
from lib.vtk_patch_scene import Patch3DScene, PatchSceneConfig


class Patch3DViewer:
    def __init__(
        self,
        scene: Patch3DScene,
        time_values,
        time_units: str | None,
        interval_ms: int,
        start: int,
        screenshot_dir: Path,
    ) -> None:
        self.scene = scene
        self.time_values = time_values
        self.time_units = time_units
        self.interval_ms = interval_ms
        self.playing = True
        n = scene.clt_series.shape[0]
        self.frame_pos = max(0, min(start, n - 1))
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        scene.configure_render_window(1280, 720, offscreen=False)
        scene.set_camera_orbit(self.frame_pos, n)
        scene.update_frame(self.frame_pos)
        scene.render()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(scene.render_window)
        self.interactor.AddObserver("KeyPressEvent", self._on_key)

    def _title(self) -> str:
        ts = wall_clock(
            None,
            self.frame_pos,
            time_values=self.time_values,
            time_units=self.time_units,
        ) or f"timestep {self.frame_pos}"
        n = self.scene.clt_series.shape[0]
        status = "" if self.playing else " (paused)"
        mode = self.scene.grid.slab_mode
        return f"Patch 3D — {mode}  {ts}  [{self.frame_pos + 1}/{n}]{status}"

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
            path = self.screenshot_dir / f"patch_3d_preview_{self.frame_pos:04d}.png"
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
    parser.add_argument("--region", choices=("west", "east", "east_core"), default="east_core")
    parser.add_argument("--slab-mode", choices=SLAB_MODES, default="clt")
    parser.add_argument("--label-source", choices=("era5", "icon"), default="era5")
    add_era5_arg(parser)
    parser.add_argument("--era5-time", help="ISO UTC hour for ERA5 snapshot")
    parser.add_argument("--z-min-m", type=float, default=1000.0)
    parser.add_argument("--center-lon", type=float, default=PATCH_CENTER_LON)
    parser.add_argument("--center-lat", type=float, default=PATCH_CENTER_LAT)
    parser.add_argument("--half-size", type=float, default=PATCH_HALF_SIZE)
    parser.add_argument("--vert-exag", type=float, default=None)
    parser.add_argument("--slab-height", type=float, default=400.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=400)
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=CLOUDS_ROOT / "output" / "frames",
    )
    args = parser.parse_args()

    lon_min, lon_max, lat_min, lat_max, center_lon, center_lat = resolve_patch_bounds(
        region=args.region,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        half_size=args.half_size,
    )
    vert_exag = args.vert_exag if args.vert_exag is not None else default_patch_vert_exag(args.region)

    nc_path = find_lonlat_nc(args.input)
    if args.slab_mode == "inversion":
        grid = load_patch_grid_inversion(
            region=args.region,
            label_source=args.label_source,
            icon_nc=nc_path,
            stride=1,
            z_min_m=args.z_min_m,
            era5_time=args.era5_time,
            max_time_steps=121,
        )
    else:
        grid = load_patch_grid(
            nc_path,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            center_lon=center_lon,
            center_lat=center_lat,
            slab_mode=args.slab_mode,
        )
    config = PatchSceneConfig(
        vert_exag=vert_exag,
        slab_height=args.slab_height,
        slab_mode=args.slab_mode,
    )
    scene = Patch3DScene(grid, config)
    viewer = Patch3DViewer(
        scene,
        grid.time_values,
        grid.time_units,
        args.interval_ms,
        args.start,
        args.screenshot_dir,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
