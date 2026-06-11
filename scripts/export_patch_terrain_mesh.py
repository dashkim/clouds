#!/usr/bin/env python3
"""Export patch terrain (and optional cloud slab) as VTK .vtp meshes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import (
    PATCH_CENTER_LAT,
    PATCH_CENTER_LON,
    PATCH_HALF_SIZE,
    default_patch_vert_exag,
    resolve_patch_bounds,
)
from lib.patch_fields import load_patch_grid_inversion
from lib.paths import CLOUDS_ROOT, add_era5_arg, add_input_arg, find_lonlat_nc
from lib.terrain_mesh import (
    build_cloud_deck_slab,
    build_cloud_slab,
    build_inversion_terrain_overlay,
    build_terrain_surface,
    load_patch_grid,
    write_vtp,
)

# Timesteps with strongest mean |Δclt| in the east patch window
EAST_KEYFRAME_TIMESTEPS = (0, 6, 12, 15, 24, 36, 48, 60, 72, 84, 96, 108, 120)


def parse_timesteps(spec: str, n_time: int) -> list[int]:
    if spec == "keyframes":
        return [t for t in EAST_KEYFRAME_TIMESTEPS if t < n_time]
    return [max(0, min(int(t.strip()), n_time - 1)) for t in spec.split(",") if t.strip()]


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
        "--output",
        type=Path,
        default=None,
        help="Terrain .vtp path (default: output/meshes/{region}_patch_terrain.vtp)",
    )
    parser.add_argument("--prefix", default=None, help="Filename prefix (default: region name)")
    parser.add_argument("--center-lon", type=float, default=PATCH_CENTER_LON)
    parser.add_argument("--center-lat", type=float, default=PATCH_CENTER_LAT)
    parser.add_argument("--half-size", type=float, default=PATCH_HALF_SIZE)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--vert-exag", type=float, default=None)
    parser.add_argument("--slab-height", type=float, default=400.0)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument(
        "--timesteps",
        default=None,
        help="Comma-separated timesteps or 'keyframes' for east-patch motion peaks",
    )
    parser.add_argument(
        "--with-cloud-slab",
        action="store_true",
        help="Export cloud slab mesh(es)",
    )
    parser.add_argument(
        "--with-inversion-mesh",
        action="store_true",
        help="Export cloud deck + inversion ridge meshes per timestep",
    )
    parser.add_argument(
        "--label-source",
        choices=("era5", "icon"),
        default="era5",
    )
    add_era5_arg(parser)
    parser.add_argument("--era5-time", help="ISO UTC hour for ERA5 snapshot")
    parser.add_argument("--z-min-m", type=float, default=1000.0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    region = args.region or "west"
    prefix = args.prefix or (f"{args.region}_patch" if args.region else "patch")
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

    output = args.output or (CLOUDS_ROOT / "output" / "meshes" / f"{prefix}_terrain.vtp")
    vert_exag = args.vert_exag if args.vert_exag is not None else default_patch_vert_exag(region)

    nc_path = find_lonlat_nc(args.input)
    grid = load_patch_grid(
        nc_path,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        center_lon=center_lon,
        center_lat=center_lat,
        load_clt=args.with_cloud_slab,
        stride=args.stride,
    )

    terrain = build_terrain_surface(grid, vert_exag)
    write_vtp(terrain, output)
    print(f"Wrote terrain mesh: {output}  ({terrain.GetNumberOfPoints()} points)")

    if args.with_cloud_slab:
        if grid.clt is None:
            raise RuntimeError("clt not loaded")
        n_time = grid.clt.shape[0]
        if args.timesteps:
            steps = parse_timesteps(args.timesteps, n_time)
        else:
            steps = [max(0, min(args.timestep, n_time - 1))]

        mesh_dir = output.parent
        for t in steps:
            cloud_path = mesh_dir / f"{prefix}_cloud_slab_t{t:03d}.vtp"
            slab = build_cloud_slab(
                grid,
                grid.clt[t],
                vert_exag=vert_exag,
                slab_height=args.slab_height,
            )
            write_vtp(slab, cloud_path)
            print(f"Wrote cloud slab: {cloud_path}  (timestep {t})")

    if args.with_inversion_mesh:
        inv_grid = load_patch_grid_inversion(
            region=region,
            label_source=args.label_source,
            icon_nc=nc_path,
            era5_grib=None,
            stride=args.stride,
            z_min_m=args.z_min_m,
            era5_time=args.era5_time,
        )
        n_time = inv_grid.overlay.shape[0]
        if args.timesteps:
            steps = parse_timesteps(args.timesteps, n_time)
        else:
            steps = [max(0, min(args.timestep, n_time - 1))]
        mesh_dir = output.parent
        for t in steps:
            deck_cover = inv_grid.cloud_deck_cover
            cover_t = deck_cover[t] if deck_cover is not None else inv_grid.clt[t]
            base_t = None
            if inv_grid.cloud_base_m is not None:
                base_t = inv_grid.cloud_base_m[t] if inv_grid.cloud_base_m.ndim == 3 else inv_grid.cloud_base_m
            deck_path = mesh_dir / f"{prefix}_cloud_deck_t{t:03d}.vtp"
            ridge_path = mesh_dir / f"{prefix}_inversion_ridge_t{t:03d}.vtp"
            deck = build_cloud_deck_slab(
                inv_grid, cover_t, base_t, vert_exag=vert_exag,
            )
            ridge = build_inversion_terrain_overlay(
                inv_grid, inv_grid.inversion[t], vert_exag=vert_exag,
            )
            write_vtp(deck, deck_path)
            write_vtp(ridge, ridge_path)
            print(f"Wrote inversion meshes: {deck_path.name}, {ridge_path.name}  (t={t})")

    print(
        f"Patch: {lon_min:.3f}–{lon_max:.3f}°E, "
        f"{lat_min:.3f}–{lat_max:.3f}°N"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
