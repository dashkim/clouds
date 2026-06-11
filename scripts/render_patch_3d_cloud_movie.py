#!/usr/bin/env python3
"""Render 3D patch movie: metric terrain mesh + pseudo-3D scalar slab."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.video_encode import encode_gif, encode_mp4
from lib.alps_region import (
    PATCH_CENTER_LAT,
    PATCH_CENTER_LON,
    PATCH_HALF_SIZE,
    default_patch_stride,
    default_patch_vert_exag,
    resolve_patch_bounds,
)
from lib.frame_overlay import FrameOverlay, annotate_frame_png
from lib.netcdf import wall_clock
from lib.patch_fields import load_patch_grid_inversion
from lib.paths import CLOUDS_ROOT, add_era5_arg, add_input_arg, find_lonlat_nc
from lib.temporal_interp import densify_patch_grid, timestamp_at_index
from lib.terrain_mesh import SLAB_MODES, load_patch_grid, patch_horizontal_span_m
from lib.vtk_patch_scene import Patch3DScene, PatchSceneConfig, _slab_bar_title

REGION_TITLES = {
    "west": "Bavarian Alps — Zugspitze region",
    "east": "Eastern Germany — Bavarian Forest (wide)",
    "east_core": "Bavarian / Bohemian Forest — mountain core",
}

SLAB_METADATA = {
    "clt": {
        "subtitle": "Simulated terrain + total cloud cover",
        "notes_suffix": "clouds: 2D clt pseudo-3D slab",
        "color_mode": "cloud",
        "default_orbit_start": -50.0,
        "frame_prefix": "3d",
        "output_name": "east_patch_3d_clouds.gif",
    },
    "pressure": {
        "subtitle": "Simulated terrain + cloud-base height (from ccb)",
        "notes_suffix": "cloud base height from ccb (Pa), masked by clt; pseudo-3D slab",
        "color_mode": "pressure",
        "default_orbit_start": 25.0,
        "frame_prefix": "3d_pressure",
        "output_name": "east_patch_3d_pressure.gif",
    },
    "wind": {
        "subtitle": "Simulated terrain + 10 m wind speed and direction",
        "notes_suffix": "10 m wind (u_10m, v_10m) on pseudo-3D slab with arrows",
        "color_mode": "wind",
        "default_orbit_start": 115.0,
        "frame_prefix": "3d_wind",
        "output_name": "east_patch_3d_wind.gif",
    },
    "inversion": {
        "subtitle": "Peaks above valley cloud deck — inversion mask",
        "notes_suffix": "horizontal cloud layer at inversion height; peaks above",
        "color_mode": "cloud",
        "default_orbit_start": -50.0,
        "default_fps": 12,
        "default_render_multiplier": 4,
        "default_cloud_time_scale": 0.35,
        "frame_prefix": "3d_inversion",
        "output_name": "east_core_patch_3d_inversion.gif",
    },
}


def _dry_run_stats(grid, *, region: str, slab_mode: str, stride: int) -> dict:
    span = patch_horizontal_span_m(grid)
    stats: dict = {
        "region": region,
        "slab_mode": slab_mode,
        "n_lon": grid.lon.size,
        "n_lat": grid.lat.size,
        "n_points": grid.lon.size * grid.lat.size,
        "z_min": float(grid.z[np.isfinite(grid.z)].min()),
        "z_max": float(grid.z[np.isfinite(grid.z)].max()),
        "span_m": round(span, 1),
        "stride": stride,
    }
    if grid.overlay is not None:
        valid = grid.overlay[np.isfinite(grid.overlay)]
        if valid.size:
            stats["overlay_min"] = float(np.nanmin(valid))
            stats["overlay_max"] = float(np.nanmax(valid))
    return stats


def render_movie(
    nc_path: Path,
    output: Path,
    *,
    region: str,
    slab_mode: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    center_lon: float,
    center_lat: float,
    vert_exag: float,
    slab_height: float,
    fps: int,
    width: int,
    height: int,
    max_frames: int | None,
    camera: str,
    orbit_deg: float,
    orbit_start_deg: float | None,
    orbit_radius_factor: float,
    camera_view: str,
    elevation_angle_deg: float,
    camera_elev_percentile: float,
    focal_elev_percentile: float,
    offscreen: bool,
    keep_frames: bool,
    dry_run: bool,
    stride: int,
    frame_prefix: str,
    annotate: bool,
    show_scalar_bars: bool,
    label_source: str = "era5",
    era5_grib: str | None = None,
    era5_time: str | None = None,
    era5_start: str | None = None,
    era5_n_frames: int | None = None,
    z_min_m: float = 800.0,
    interp_steps: int = 1,
    render_multiplier: float = 1.0,
    cloud_time_scale: float = 1.0,
) -> Path:
    meta = SLAB_METADATA[slab_mode]
    orbit_start = (
        orbit_start_deg
        if orbit_start_deg is not None
        else meta["default_orbit_start"]
    )

    if slab_mode == "inversion":
        grid = load_patch_grid_inversion(
            region=region,
            label_source=label_source,
            icon_nc=nc_path,
            era5_grib=Path(era5_grib) if era5_grib else None,
            stride=stride,
            z_min_m=z_min_m,
            era5_time=era5_time,
            era5_start=era5_start,
            era5_n_frames=era5_n_frames,
            max_time_steps=max_frames,
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
            slab_mode=slab_mode,
            stride=stride,
        )

    if dry_run:
        print("Dry run — patch stats:", _dry_run_stats(
            grid, region=region, slab_mode=slab_mode, stride=stride,
        ))
        return output

    if interp_steps > 1:
        grid = densify_patch_grid(grid, interp_steps)
        print(f"Temporal upsampling: {interp_steps} steps/hour → {grid.overlay.shape[0]} frames")

    n_cloud = grid.overlay.shape[0] if grid.overlay is not None else 0
    if max_frames is not None:
        n_cloud = min(n_cloud, max_frames)

    n_render = max(n_cloud, int(round(n_cloud * render_multiplier)))
    decouple_time = render_multiplier > 1.0 or cloud_time_scale < 1.0
    if decouple_time:
        print(
            f"Render: {n_render} frames @ {fps} fps  "
            f"(cloud keyframes={n_cloud}, cloud_time_scale={cloud_time_scale:.2f})"
        )

    vtk_bars = show_scalar_bars and not annotate
    config = PatchSceneConfig(
        vert_exag=vert_exag,
        slab_height=slab_height,
        slab_mode=slab_mode,
        orbit_deg=orbit_deg,
        orbit_start_deg=orbit_start,
        orbit_radius_factor=orbit_radius_factor,
        camera_view=camera_view,
        elevation_angle_deg=elevation_angle_deg,
        camera_elev_percentile=camera_elev_percentile,
        focal_elev_percentile=focal_elev_percentile,
        show_scalar_bars=vtk_bars,
    )
    scene = Patch3DScene(
        grid,
        config,
        region_title=REGION_TITLES.get(region, "Alps patch"),
    )
    scene.configure_render_window(width, height, offscreen=offscreen)

    region_label = (
        f"{lon_min:.2f}–{lon_max:.2f}°E, {lat_min:.2f}–{lat_max:.2f}°N"
    )
    label_note = f"labels: {label_source}" if slab_mode == "inversion" else "ICON HD(CP)² 26 Apr 2013"
    notes = (
        f"{label_note}  ·  vertical exaggeration ×{vert_exag:.0f}  ·  "
        f"{meta['notes_suffix']}"
    )

    frame_dir = CLOUDS_ROOT / "output" / "frames" / frame_prefix
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frame_pattern = f"{frame_prefix}_%04d.png"

    for r in range(n_render):
        cam_frac = r / max(n_render - 1, 1)
        cloud_t = cam_frac * cloud_time_scale * max(n_cloud - 1, 0)

        if decouple_time:
            scene.update_frame_interp(cloud_t)
            scene.set_camera_orbit(r, n_render, mode=camera)
        else:
            scene.update_frame(int(round(cloud_t)))
            scene.set_camera_orbit(r, n_render, mode=camera)

        scene.render()
        frame_path = frame_dir / (frame_pattern % r)
        scene.screenshot_png(str(frame_path))

        if annotate:
            if slab_mode == "inversion" and grid.era5_frame_times:
                ts = timestamp_at_index(grid.era5_frame_times, cloud_t) or f"timestep {cloud_t:.1f}"
            else:
                ts = wall_clock(
                    None,
                    int(round(cloud_t)),
                    time_values=grid.time_values,
                    time_units=grid.time_units,
                ) or f"timestep {cloud_t:.1f}"
            annotate_frame_png(
                frame_path,
                FrameOverlay(
                    title=REGION_TITLES.get(region, "Alps patch"),
                    subtitle=meta["subtitle"],
                    timestamp=ts,
                    frame_label=f"Frame {r + 1} / {n_render}",
                    region_label=region_label,
                    notes=notes,
                    elev_vmin=scene.elev_vmin,
                    elev_vmax=scene.elev_vmax,
                    slab_label=_slab_bar_title(slab_mode),
                    slab_vmin=scene.config.slab_vmin,
                    slab_vmax=scene.config.slab_vmax,
                    slab_color_mode=meta["color_mode"],
                ),
            )

        frame_paths.append(frame_path)
        if annotate:
            print(f"Rendered {r + 1}/{n_render}  {ts}")
        else:
            print(f"Rendered {r + 1}/{n_render}  cloud_t={cloud_t:.2f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4" and encode_mp4(frame_dir, frame_pattern, output, fps):
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
        "--region",
        choices=("west", "east", "east_core"),
        default="east_core",
        help="Preset patch (east_core = zoomed mountain core)",
    )
    parser.add_argument(
        "--slab-mode",
        choices=SLAB_MODES,
        default="clt",
        help="Scalar field on pseudo-3D slab: clt, pressure, wind, inversion",
    )
    parser.add_argument(
        "--label-source",
        choices=("era5", "icon"),
        default="era5",
        help="Inversion mask source (inversion slab-mode only)",
    )
    add_era5_arg(parser)
    parser.add_argument(
        "--era5-time",
        help="ISO UTC hour for single-frame ERA5 snapshot (e.g. 2013-04-26T20:00)",
    )
    parser.add_argument("--era5-start", help="ISO UTC start hour for ERA5 animation window")
    parser.add_argument("--era5-frames", type=int, default=8, help="Hours to animate from --era5-start")
    parser.add_argument(
        "--interp-steps",
        type=int,
        default=1,
        help="Blend frames between each hour (4 → 3 transitions per gap; good for ERA5)",
    )
    parser.add_argument("--z-min-m", type=float, default=800.0, help="Ridge elevation cutoff (m)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--center-lon", type=float, default=PATCH_CENTER_LON)
    parser.add_argument("--center-lat", type=float, default=PATCH_CENTER_LAT)
    parser.add_argument("--half-size", type=float, default=PATCH_HALF_SIZE)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--vert-exag", type=float, default=None)
    parser.add_argument("--slab-height", type=float, default=400.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="GIF frame rate (default: 12 inversion, 8 otherwise)",
    )
    parser.add_argument(
        "--render-multiplier",
        type=float,
        default=None,
        help="Extra render frames for smooth orbit (default 4 for inversion)",
    )
    parser.add_argument(
        "--cloud-time-scale",
        type=float,
        default=None,
        help="Fraction of cloud timeline traversed per full orbit (default 0.35 inversion)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", choices=("orbit", "static"), default="orbit")
    parser.add_argument("--orbit-deg", type=float, default=40.0, help="Total orbit sweep (lower = slower)")
    parser.add_argument(
        "--orbit-start-deg",
        type=float,
        default=None,
        help="Initial camera azimuth (defaults per slab mode)",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=None,
        help="Camera distance factor (lower = zoom in; default 0.88 east_core, 1.15 otherwise)",
    )
    parser.add_argument(
        "--camera-view",
        choices=("horizon", "oblique"),
        default="oblique",
        help="horizon = look across at ridges; oblique = look down from above",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=24.0,
        help="Oblique mode: camera pitch above horizontal",
    )
    parser.add_argument(
        "--camera-elev-percentile",
        type=float,
        default=24.0,
        help="Horizon mode: eyepoint height on terrain (lower = flatter view)",
    )
    parser.add_argument(
        "--focal-elev-percentile",
        type=float,
        default=71.0,
        help="Horizon mode: look-at height on terrain (higher = aim at peaks)",
    )
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--no-scalar-bars", action="store_true")
    parser.add_argument("--use-screen", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lon_min, lon_max, lat_min, lat_max, center_lon, center_lat = resolve_patch_bounds(
        region=args.region if args.lon_min is None else None,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        half_size=args.half_size,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )

    stride = args.stride if args.stride is not None else default_patch_stride(args.region)
    vert_exag = args.vert_exag if args.vert_exag is not None else default_patch_vert_exag(args.region)
    orbit_radius = (
        args.orbit_radius
        if args.orbit_radius is not None
        else (0.88 if args.region == "east_core" else 1.15)
    )

    slab_meta = SLAB_METADATA[args.slab_mode]
    prefix = f"{args.region}_patch"
    if args.output is None:
        if args.region == "east_core":
            output = CLOUDS_ROOT / "output" / "movies" / slab_meta["output_name"]
        else:
            output = CLOUDS_ROOT / "output" / "movies" / f"{prefix}_{slab_meta['frame_prefix']}.gif"
    else:
        output = args.output

    frame_prefix = f"{prefix}_{slab_meta['frame_prefix']}"
    fps = args.fps if args.fps is not None else slab_meta.get("default_fps", 8)
    render_multiplier = (
        args.render_multiplier
        if args.render_multiplier is not None
        else slab_meta.get("default_render_multiplier", 1.0)
    )
    cloud_time_scale = (
        args.cloud_time_scale
        if args.cloud_time_scale is not None
        else slab_meta.get("default_cloud_time_scale", 1.0)
    )

    nc_path = find_lonlat_nc(args.input)
    out = render_movie(
        nc_path,
        output,
        region=args.region,
        slab_mode=args.slab_mode,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        center_lon=center_lon,
        center_lat=center_lat,
        vert_exag=vert_exag,
        slab_height=args.slab_height,
        fps=fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        camera=args.camera,
        orbit_deg=args.orbit_deg,
        orbit_start_deg=args.orbit_start_deg,
        orbit_radius_factor=orbit_radius,
        camera_view=args.camera_view,
        elevation_angle_deg=args.elevation_deg,
        camera_elev_percentile=args.camera_elev_percentile,
        focal_elev_percentile=args.focal_elev_percentile,
        offscreen=not args.use_screen,
        keep_frames=args.keep_frames,
        dry_run=args.dry_run,
        stride=stride,
        frame_prefix=frame_prefix,
        annotate=not args.no_annotate,
        show_scalar_bars=not args.no_scalar_bars,
        label_source=args.label_source,
        era5_grib=args.era5_grib,
        era5_time=args.era5_time,
        era5_start=args.era5_start,
        era5_n_frames=args.era5_frames,
        z_min_m=args.z_min_m,
        interp_steps=args.interp_steps,
        render_multiplier=render_multiplier,
        cloud_time_scale=cloud_time_scale,
    )
    if not args.dry_run:
        print(f"Done: {out}")
        print(f"Patch: {lon_min:.3f}–{lon_max:.3f}°E, {lat_min:.3f}–{lat_max:.3f}°N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
