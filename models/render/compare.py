#!/usr/bin/env python3
"""Render 3D movie comparing actual vs predicted inversion fog decks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.alps_region import default_patch_stride, default_patch_vert_exag, resolve_patch_bounds
from lib.patch_fields import load_patch_fields, patch_fields_to_grid
from lib.paths import CLOUDS_ROOT, MODELS_ARTIFACTS_DIR, add_era5_arg, add_input_arg, find_lonlat_nc
from lib.terrain_mesh import PatchGrid
from scripts.render_patch_3d_cloud_movie import REGION_TITLES

# Strong inversion window (matches east_core_patch_3d_inversion.gif start).
DEFAULT_ERA5_START = "2013-01-01T18:00"
DEFAULT_ERA5_FRAMES = 30


def _match_time_indices(npz_times: np.ndarray, frame_times: list[str]) -> list[int]:
    ts_npz = pd.to_datetime(npz_times)
    indices: list[int] = []
    for ft in frame_times:
        target = pd.Timestamp(ft)
        idx = int(np.argmin(np.abs(ts_npz - target)))
        indices.append(idx)
    return indices


def load_compare_patch_grid(
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
    """Load ERA5 cloud context + ML predicted masks for deck heights."""
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
    grid.slab_mode = "inversion_compare"

    data = np.load(predictions_npz, allow_pickle=False)
    npz_times = data["times"]
    predicted_all = data["predicted"]

    if fields.era5_frame_times is None:
        raise ValueError("era5_start / era5_n_frames required for compare movie")

    idx = _match_time_indices(npz_times, fields.era5_frame_times)
    predicted = predicted_all[idx].astype(np.float32)

    grid.overlay = fields.inversion
    grid.inversion = fields.inversion
    grid.predicted_inversion = predicted
    grid.comparison_field = None
    return grid


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
    parser.add_argument(
        "--era5-start",
        default=DEFAULT_ERA5_START,
        help=f"ISO UTC start (default: {DEFAULT_ERA5_START})",
    )
    parser.add_argument(
        "--era5-frames",
        type=int,
        default=DEFAULT_ERA5_FRAMES,
        help=f"Hours to animate (default: {DEFAULT_ERA5_FRAMES})",
    )
    parser.add_argument("--z-min-m", type=float, default=800.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--interp-steps", type=int, default=6, help="Sub-frames per hour")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--render-multiplier",
        type=float,
        default=1.0,
        help="Extra render frames vs cloud frames (1 = one frame per timestep)",
    )
    parser.add_argument(
        "--cloud-time-scale",
        type=float,
        default=1.0,
        help="Fraction of timeline shown (1 = full window)",
    )
    parser.add_argument("--camera", choices=("orbit", "static"), default="orbit")
    parser.add_argument("--elevation-deg", type=float, default=24.0)
    parser.add_argument("--orbit-deg", type=float, default=40.0)
    parser.add_argument("--orbit-start-deg", type=float, default=-50.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--use-screen", action="store_true")
    args = parser.parse_args()

    lon_min, lon_max, lat_min, lat_max, center_lon, center_lat = resolve_patch_bounds(
        region=args.region,
    )
    stride = args.stride if args.stride is not None else default_patch_stride(args.region)
    vert_exag = default_patch_vert_exag(args.region)

    nc_path = find_lonlat_nc(args.input)
    grid = load_compare_patch_grid(
        predictions_npz=args.predictions,
        region=args.region,
        icon_nc=nc_path,
        era5_grib=Path(args.era5_grib) if args.era5_grib else None,
        stride=stride,
        z_min_m=args.z_min_m,
        era5_start=args.era5_start,
        era5_n_frames=args.era5_frames,
    )

    if args.output is None:
        output = CLOUDS_ROOT / "output" / "movies" / f"{args.region}_inversion_compare.gif"
    else:
        output = args.output

    from lib.temporal_interp import densify_patch_grid

    if args.interp_steps > 1:
        grid = densify_patch_grid(grid, args.interp_steps)

    from lib.frame_overlay import FrameOverlay, annotate_frame_png
    from lib.netcdf import wall_clock
    from lib.temporal_interp import timestamp_at_index
    from lib.vtk_patch_scene import Patch3DScene, PatchSceneConfig
    from lib.video_encode import encode_gif, encode_mp4

    config = PatchSceneConfig(
        vert_exag=vert_exag,
        slab_mode="inversion_compare",
        orbit_deg=args.orbit_deg,
        orbit_start_deg=args.orbit_start_deg,
        orbit_radius_factor=0.88 if args.region == "east_core" else 1.15,
        camera_view="oblique",
        elevation_angle_deg=args.elevation_deg,
        show_scalar_bars=False,
        compare_deck_opacity=0.72,
        terrain_opacity_compare=0.92,
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
    notes = "Cyan = actual fog deck  ·  Orange = ML predicted deck"

    frame_dir = CLOUDS_ROOT / "output" / "frames" / f"{args.region}_inversion_compare"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    pattern = f"{args.region}_compare_%04d.png"

    print(
        f"Rendering {n_render} frames @ {args.fps} fps  "
        f"({n_cloud} timesteps, {args.era5_start} + {args.era5_frames}h)"
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

        annotate_frame_png(
            frame_path,
            FrameOverlay(
                title=REGION_TITLES.get(args.region, "Alps patch"),
                subtitle="Actual vs predicted inversion fog decks",
                timestamp=ts,
                frame_label=f"Frame {r + 1} / {n_render}",
                region_label=region_label,
                notes=notes,
                elev_vmin=scene.elev_vmin,
                elev_vmax=scene.elev_vmax,
                slab_label="",
                slab_vmin=0.0,
                slab_vmax=1.0,
                slab_color_mode="cloud",
                legend_mode="compare",
            ),
        )
        frame_paths.append(frame_path)
        if r == 0 or r == n_render - 1 or (r + 1) % max(1, n_render // 10) == 0:
            print(f"Rendered {r + 1}/{n_render}  {ts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4" and encode_mp4(frame_dir, pattern, output, args.fps):
        print(f"Wrote {output}")
    else:
        if output.suffix.lower() == ".mp4":
            output = output.with_suffix(".gif")
        encode_gif(frame_paths, output, args.fps)
        print(f"Wrote {output}")

    if not args.keep_frames:
        for p in frame_paths:
            p.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
