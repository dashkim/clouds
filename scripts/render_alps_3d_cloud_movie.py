#!/usr/bin/env python3
"""Render 3D Bavarian Alps cloud movie: extruded terrain + pseudo-3D clt slab."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.video_encode import encode_gif, encode_mp4
from lib.alps_region import (
    DEFAULT_ELEV_MIN,
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    crop_indices,
    estimate_crop_shape,
    load_crop_multi,
)
from lib.netcdf import wall_clock
from lib.paths import CLOUDS_ROOT, add_input_arg, find_lonlat_nc
from lib.vtk_alps_scene import Alps3DScene, AlpsSceneConfig


def render_movie(
    nc_path: Path,
    output: Path,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    elev_min: float,
    vert_exag: float,
    slab_height: float,
    stride: int,
    fps: int,
    width: int,
    height: int,
    max_frames: int | None,
    camera: str,
    orbit_deg: float,
    offscreen: bool,
    keep_frames: bool,
    dry_run: bool,
) -> Path:
    if dry_run:
        stats = estimate_crop_shape(
            nc_path, lon_min, lon_max, lat_min, lat_max, elev_min, stride,
        )
        print("Dry run — crop stats:", stats)
        return output

    with nc.Dataset(nc_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            lon_min, lon_max, lat_min, lat_max,
        )

    fields, meta = load_crop_multi(
        nc_path, lon_sl, lat_sl, ["clt"], stride=stride,
    )
    n_time = meta.n_time
    if max_frames is not None:
        n_time = min(n_time, max_frames)

    config = AlpsSceneConfig(
        elev_min=elev_min,
        vert_exag=vert_exag,
        slab_height=slab_height,
        orbit_deg=orbit_deg,
    )
    scene = Alps3DScene(meta.lon, meta.lat, meta.z, fields["clt"], config)
    scene.configure_render_window(width, height, offscreen=offscreen)

    frame_dir = CLOUDS_ROOT / "output" / "frames" / "alps_3d"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    for t in range(n_time):
        scene.update_frame(t)
        scene.set_camera_orbit(t, n_time, mode=camera)
        scene.render()
        frame_path = frame_dir / f"alps_3d_{t:04d}.png"
        scene.screenshot_png(str(frame_path))
        frame_paths.append(frame_path)
        ts = wall_clock(
            meta.time_var,
            t,
            time_values=meta.time_values,
            time_units=meta.time_units,
        ) or f"frame {t}"
        print(f"Rendered {t + 1}/{n_time}  {ts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4" and encode_mp4(frame_dir, "alps_3d_%04d.png", output, fps):
        print(f"Wrote {output}")
    else:
        if output.suffix.lower() == ".mp4":
            output = output.with_suffix(".gif")
        encode_gif(frame_paths, output, fps)
        print(f"Wrote {output} (ffmpeg unavailable or mp4 failed)")

    if not keep_frames:
        for p in frame_paths:
            p.unlink(missing_ok=True)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    parser.add_argument(
        "--output",
        type=Path,
        default=CLOUDS_ROOT / "output" / "movies" / "alps_3d_clouds.mp4",
    )
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--elev-min", type=float, default=DEFAULT_ELEV_MIN)
    parser.add_argument("--vert-exag", type=float, default=3.5)
    parser.add_argument("--slab-height", type=float, default=600.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", choices=("orbit", "static"), default="orbit")
    parser.add_argument("--orbit-deg", type=float, default=120.0)
    parser.add_argument(
        "--use-screen",
        action="store_true",
        help="Use on-screen render window (fallback if offscreen GL fails)",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    nc_path = find_lonlat_nc(args.input)
    out = render_movie(
        nc_path,
        args.output,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        elev_min=args.elev_min,
        vert_exag=args.vert_exag,
        slab_height=args.slab_height,
        stride=args.stride,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        camera=args.camera,
        orbit_deg=args.orbit_deg,
        offscreen=not args.use_screen,
        keep_frames=args.keep_frames,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
