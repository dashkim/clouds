#!/usr/bin/env python3
"""Render 3D movies: ML predicted deck + one Ridge input feature each (LCC, T2m, wind)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.alps_region import default_patch_stride, default_patch_vert_exag, resolve_patch_bounds
from lib.patch_fields import load_patch_fields, patch_fields_to_grid
from lib.paths import CLOUDS_ROOT, MODELS_ARTIFACTS_DIR, add_era5_arg, add_input_arg, find_lonlat_nc
from lib.terrain_mesh import PatchGrid
from lib.temporal_interp import densify_patch_grid, lerp_time_series, timestamp_at_index
from models.inversion.viz import (
    format_driver_notes,
    load_era5_gridded_features,
    patch_mean_features,
    top_spatial_drivers,
)
from models.render.compare import (
    DEFAULT_ERA5_FRAMES,
    DEFAULT_ERA5_START,
    _match_time_indices,
)
from scripts.render_patch_3d_cloud_movie import REGION_TITLES
from lib.video_encode import encode_gif, encode_mp4

DEFAULT_MODEL = MODELS_ARTIFACTS_DIR / "east_core_ridge.joblib"

FEATURE_KINDS = ("lcc", "t2m", "wind")

FEATURE_META = {
    "lcc": {
        "subtitle": "Predicted inversion deck + low cloud cover (LCC)",
        "suffix": "lcc",
        "driver_key": "lcc",
    },
    "t2m": {
        "subtitle": "Predicted inversion deck + 2 m temperature",
        "suffix": "t2m",
        "driver_key": "t2m",
    },
    "wind": {
        "subtitle": "Predicted inversion deck + 10 m wind",
        "suffix": "wind",
        "driver_key": "v10",
    },
}


def load_feature_overlay_grid(
    *,
    predictions_npz: Path,
    region: str = "east_core",
    label_source: str = "era5",
    icon_nc: Path | None = None,
    era5_grib: Path | None = None,
    stride: int | None = None,
    z_min_m: float = 800.0,
    era5_start: str | None = None,
    era5_n_frames: int | None = None,
) -> PatchGrid:
    era5_start = era5_start or DEFAULT_ERA5_START
    era5_n_frames = era5_n_frames or DEFAULT_ERA5_FRAMES

    fields = load_patch_fields(
        region=region,
        label_source=label_source,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        stride=stride,
        z_min_m=z_min_m,
        era5_start=era5_start,
        era5_n_frames=era5_n_frames,
    )
    grid = patch_fields_to_grid(fields, slab_mode="inversion")
    grid.slab_mode = "feature_overlay"

    extra = load_era5_gridded_features(
        region=region,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        stride=stride,
        era5_start=era5_start,
        era5_n_frames=era5_n_frames,
        variables=("t2m", "u10", "v10"),
    )
    grid.feature_t2m = extra["t2m"]
    grid.u_wind = extra["u10"]
    grid.v_wind = extra["v10"]

    data = np.load(predictions_npz, allow_pickle=False)
    if fields.era5_frame_times is None:
        raise ValueError("era5_start / era5_n_frames required for feature overlay movie")

    idx = _match_time_indices(data["times"], fields.era5_frame_times)
    grid.predicted_inversion = data["predicted"][idx].astype(np.float32)
    grid.overlay = fields.inversion
    grid.inversion = fields.inversion
    return grid


def render_feature_movie(
    *,
    grid: PatchGrid,
    feature_kind: str,
    drivers: list[tuple[str, float]],
    args: argparse.Namespace,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    vert_exag: float,
) -> Path:
    from lib.frame_overlay import FrameOverlay, annotate_frame_png
    from lib.netcdf import wall_clock
    from lib.vtk_patch_scene import Patch3DScene, PatchSceneConfig

    meta = FEATURE_META[feature_kind]
    if args.output is not None and args.feature != "all":
        output = args.output
    else:
        output = (
            CLOUDS_ROOT / "output" / "movies"
            / f"{args.region}_feature_overlay_{meta['suffix']}.gif"
        )

    config = PatchSceneConfig(
        vert_exag=vert_exag,
        slab_mode="feature_overlay",
        feature_overlay_kind=feature_kind,
        orbit_deg=args.orbit_deg,
        orbit_start_deg=args.orbit_start_deg,
        orbit_radius_factor=0.88 if args.region == "east_core" else 1.15,
        camera_view="oblique",
        elevation_angle_deg=args.elevation_deg,
        show_scalar_bars=False,
        compare_deck_opacity=0.78,
    )
    scene = Patch3DScene(
        grid,
        config,
        region_title=REGION_TITLES.get(args.region, "Alps patch"),
    )
    scene.configure_render_window(args.width, args.height, offscreen=not args.use_screen)

    n_cloud = grid.overlay.shape[0]
    n_render = max(n_cloud, int(round(n_cloud * args.render_multiplier)))
    region_label = f"{lon_min:.2f}–{lon_max:.2f}°E, {lat_min:.2f}–{lat_max:.2f}°N"

    frame_dir = CLOUDS_ROOT / "output" / "frames" / f"{args.region}_feature_{meta['suffix']}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    pattern = f"{args.region}_feature_{meta['suffix']}_%04d.png"

    print(
        f"[{feature_kind}] Rendering {n_render} frames @ {args.fps} fps  "
        f"({n_cloud} timesteps, scale {scene.feature_vmin:g}–{scene.feature_vmax:g})"
    )

    for r in range(n_render):
        cam_frac = r / max(n_render - 1, 1)
        cloud_t = cam_frac * args.cloud_time_scale * max(n_cloud - 1, 0)
        scene.update_frame_interp(cloud_t)
        scene.set_camera_orbit(r, n_render, mode=args.camera)
        scene.render()
        frame_path = frame_dir / (pattern % r)
        scene.screenshot_png(str(frame_path))

        if grid.era5_frame_times:
            ts = timestamp_at_index(grid.era5_frame_times, cloud_t) or f"timestep {cloud_t:.1f}"
        else:
            ts = wall_clock(None, int(round(cloud_t))) or f"timestep {cloud_t:.1f}"

        lcc = lerp_time_series(grid.cloud_deck_cover, cloud_t)
        t2m = lerp_time_series(grid.feature_t2m, cloud_t)
        u10 = lerp_time_series(grid.u_wind, cloud_t)
        v10 = lerp_time_series(grid.v_wind, cloud_t)
        means = patch_mean_features(lcc=lcc, t2m=t2m, u10=u10, v10=v10)
        notes = (
            f"Orange = ML predicted deck  ·  terrain color = {scene.feature_label} "
            f"(fixed scale {scene.feature_vmin:g}–{scene.feature_vmax:g})  ·  "
            f"{format_driver_notes(drivers, patch_means=means)}"
        )

        annotate_frame_png(
            frame_path,
            FrameOverlay(
                title=REGION_TITLES.get(args.region, "Alps patch"),
                subtitle=meta["subtitle"],
                timestamp=ts,
                frame_label=f"Frame {r + 1} / {n_render}",
                region_label=region_label,
                notes=notes,
                elev_vmin=scene.elev_vmin,
                elev_vmax=scene.elev_vmax,
                slab_label=scene.feature_label,
                slab_vmin=scene.feature_vmin,
                slab_vmax=scene.feature_vmax,
                slab_color_mode=scene.feature_color_mode,
                legend_mode="feature_single",
            ),
        )
        frame_paths.append(frame_path)
        if r == 0 or r == n_render - 1 or (r + 1) % max(1, n_render // 10) == 0:
            print(f"  [{feature_kind}] {r + 1}/{n_render}  {ts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4" and encode_mp4(frame_dir, pattern, output, args.fps):
        print(f"  Wrote {output}")
    else:
        if output.suffix.lower() == ".mp4":
            output = output.with_suffix(".gif")
        encode_gif(frame_paths, output, args.fps)
        print(f"  Wrote {output}")

    if not args.keep_frames:
        for p in frame_paths:
            p.unlink(missing_ok=True)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    add_era5_arg(parser)
    parser.add_argument("--region", default="east_core", choices=("west", "east", "east_core"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=MODELS_ARTIFACTS_DIR / "east_core_predicted_masks.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--feature",
        choices=(*FEATURE_KINDS, "all"),
        default="all",
        help="Which feature movie to render (default: all three)",
    )
    parser.add_argument("--era5-start", default=DEFAULT_ERA5_START)
    parser.add_argument("--era5-frames", type=int, default=DEFAULT_ERA5_FRAMES)
    parser.add_argument("--z-min-m", type=float, default=800.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--interp-steps", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render-multiplier", type=float, default=1.0)
    parser.add_argument("--cloud-time-scale", type=float, default=1.0)
    parser.add_argument("--camera", choices=("orbit", "static"), default="orbit")
    parser.add_argument("--elevation-deg", type=float, default=24.0)
    parser.add_argument("--orbit-deg", type=float, default=40.0)
    parser.add_argument("--orbit-start-deg", type=float, default=-50.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--use-screen", action="store_true")
    args = parser.parse_args()

    lon_min, lon_max, lat_min, lat_max, _, _ = resolve_patch_bounds(region=args.region)
    stride = args.stride if args.stride is not None else default_patch_stride(args.region)
    vert_exag = default_patch_vert_exag(args.region)

    nc_path = find_lonlat_nc(args.input)
    grid = load_feature_overlay_grid(
        predictions_npz=args.predictions,
        region=args.region,
        icon_nc=nc_path,
        era5_grib=Path(args.era5_grib) if args.era5_grib else None,
        stride=stride,
        z_min_m=args.z_min_m,
        era5_start=args.era5_start,
        era5_n_frames=args.era5_frames,
    )
    if args.interp_steps > 1:
        grid = densify_patch_grid(grid, args.interp_steps)

    drivers = top_spatial_drivers(args.model, n=3)
    print(f"Top spatial Ridge drivers: {format_driver_notes(drivers)}")

    kinds = FEATURE_KINDS if args.feature == "all" else (args.feature,)
    outputs: list[Path] = []
    for kind in kinds:
        outputs.append(
            render_feature_movie(
                grid=grid,
                feature_kind=kind,
                drivers=drivers,
                args=args,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                vert_exag=vert_exag,
            )
        )

    print("Done:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
