#!/usr/bin/env python3
"""Render a cloud-cover movie cropped to the Bavarian Alps region."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import (
    RECT_LAT_MAX,
    RECT_LAT_MIN,
    RECT_LON_MAX,
    RECT_LON_MIN,
    crop_indices,
    load_crop,
)
from lib.netcdf import wall_clock
from lib.paths import CLOUDS_ROOT, add_input_arg, find_lonlat_nc


def try_ffmpeg_writer(fps: int):
    if shutil.which("ffmpeg") is None:
        return None
    try:
        from matplotlib.animation import FFMpegWriter
        return FFMpegWriter(fps=fps, bitrate=4000)
    except Exception:
        return None


def render_movie(
    nc_path: Path,
    output: Path,
    var_name: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    fps: int,
    dpi: int,
) -> Path:
    with nc.Dataset(nc_path, "r") as ds:
        lon_full = ds.variables["lon"][:]
        lat_full = ds.variables["lat"][:]

    lon_sl, lat_sl = crop_indices(lon_full, lat_full, lon_min, lon_max, lat_min, lat_max)
    lon, lat, z, field, time_var, n_time = load_crop(nc_path, lon_sl, lat_sl, var_name)

    lon2d, lat2d = np.meshgrid(lon, lat)
    z_valid = z[np.isfinite(z)]
    if z_valid.size == 0:
        raise ValueError("No valid terrain in crop region.")

    vmin = float(np.nanpercentile(field, 2))
    vmax = float(np.nanpercentile(field, 98))
    if vmin >= vmax:
        vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    terrain = ax.pcolormesh(
        lon2d, lat2d, z,
        cmap="terrain",
        shading="auto",
        vmin=float(np.nanpercentile(z_valid, 5)),
        vmax=float(np.nanpercentile(z_valid, 99)),
    )
    cbar_t = fig.colorbar(terrain, ax=ax, fraction=0.035, pad=0.02)
    cbar_t.set_label("Elevation (m)", color="white")
    cbar_t.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_t.ax.yaxis.get_ticklabels(), color="white")

    cloud = ax.pcolormesh(
        lon2d, lat2d, field[0],
        cmap="Blues",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        alpha=0.65,
    )
    cbar_c = fig.colorbar(cloud, ax=ax, fraction=0.035, pad=0.08)
    cbar_c.set_label(f"{var_name} (%)", color="white")
    cbar_c.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_c.ax.yaxis.get_ticklabels(), color="white")

    title = ax.set_title("", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Longitude (°E)", color="white")
    ax.set_ylabel("Latitude (°N)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())

    region = (
        f"Bavarian Alps  ({lon_min:.1f}–{lon_max:.1f}°E, "
        f"{lat_min:.1f}–{lat_max:.1f}°N)"
    )

    def update(frame: int):
        cloud.set_array(field[frame].ravel())
        ts = wall_clock(time_var, frame) or f"timestep {frame}"
        title.set_text(f"{region}\n{var_name} total cloud cover  ·  {ts}  ·  frame {frame + 1}/{n_time}")
        return cloud, title

    anim = FuncAnimation(fig, update, frames=n_time, blit=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = try_ffmpeg_writer(fps)
    if writer is not None and output.suffix.lower() == ".mp4":
        anim.save(str(output), writer=writer, dpi=dpi)
    else:
        if output.suffix.lower() == ".mp4":
            output = output.with_suffix(".gif")
        anim.save(str(output), writer=PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    parser.add_argument("--var", default="clt", help="Field to animate (default: clt)")
    parser.add_argument(
        "--output",
        type=Path,
        default=CLOUDS_ROOT / "output" / "movies" / "alps_clouds.mp4",
    )
    parser.add_argument("--lon-min", type=float, default=RECT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=RECT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=RECT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=RECT_LAT_MAX)
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    nc_path = find_lonlat_nc(args.input)
    out = render_movie(
        nc_path,
        args.output,
        args.var,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
        args.fps,
        args.dpi,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
