#!/usr/bin/env python3
"""Render cloud movie masked to irregular Alpine terrain (elevation threshold)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import (
    DEFAULT_ELEV_MIN,
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    alps_mask,
    crop_indices,
    load_crop,
    tight_extent,
)
from lib.netcdf import wall_clock
from lib.paths import CLOUDS_ROOT, add_input_arg, find_lonlat_nc


def terrain_rgb(z: np.ndarray, mask: np.ndarray, vert_exag: float, bg: tuple[float, float, float]) -> np.ndarray:
    z_ma = np.ma.masked_where(~mask, z)
    z_valid = z[mask]
    vmin = float(np.percentile(z_valid, 8))
    vmax = float(np.percentile(z_valid, 99.5))
    ls = LightSource(azdeg=315, altdeg=42)
    rgb = ls.shade(
        z_ma,
        cmap=plt.cm.terrain,
        vert_exag=vert_exag,
        blend_mode="soft",
        vmin=vmin,
        vmax=vmax,
    )
    rgb[~mask, 0] = bg[0]
    rgb[~mask, 1] = bg[1]
    rgb[~mask, 2] = bg[2]
    if rgb.shape[-1] == 4:
        rgb[~mask, 3] = 1.0
    return rgb


def cloud_rgba(clt: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """RGBA cloud layer: white-blue where cloudy, transparent elsewhere."""
    norm = np.clip((clt - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    alpha = np.where(mask, norm * 0.85, 0.0)
    r = np.where(mask, 0.75 + 0.25 * norm, 0.0)
    g = np.where(mask, 0.82 + 0.15 * norm, 0.0)
    b = np.where(mask, 0.95, 0.0)
    return np.dstack([r, g, b, alpha]).astype(np.float32)


def try_ffmpeg_writer(fps: int):
    if shutil.which("ffmpeg") is None:
        return None
    try:
        from matplotlib.animation import FFMpegWriter
        return FFMpegWriter(fps=fps, bitrate=5000)
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
    elev_min: float,
    vert_exag: float,
    pad_deg: float,
    fps: int,
    dpi: int,
) -> Path:
    with nc.Dataset(nc_path, "r") as ds:
        lon_full = ds.variables["lon"][:]
        lat_full = ds.variables["lat"][:]

    lon_sl, lat_sl = crop_indices(lon_full, lat_full, lon_min, lon_max, lat_min, lat_max)
    lon, lat, z, field, time_var, n_time = load_crop(nc_path, lon_sl, lat_sl, var_name)

    mask = alps_mask(z, elev_min)
    if not np.any(mask):
        raise ValueError(f"No pixels at or above {elev_min} m in fetch region.")

    extent = tight_extent(lon, lat, mask, pad_deg)
    bg = "#070b14"
    bg_rgb = (0.027, 0.043, 0.078)
    terrain = terrain_rgb(z, mask, vert_exag, bg_rgb)

    masked_field = field[:, mask]
    vmin = float(np.nanpercentile(masked_field, 5))
    vmax = float(np.nanpercentile(masked_field, 95))
    if vmin >= vmax:
        vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(11, 10), facecolor=bg)
    ax.set_facecolor(bg)

    ax.imshow(
        terrain,
        extent=(lon.min(), lon.max(), lat.min(), lat.max()),
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
    )
    im_cloud = ax.imshow(
        cloud_rgba(field[0], mask, vmin, vmax),
        extent=(lon.min(), lon.max(), lat.min(), lat.max()),
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
    )

    z_outline = np.where(mask, z, np.nan)
    ax.contour(
        lon, lat, z_outline,
        levels=[elev_min],
        colors=["#ffffff"],
        linewidths=0.6,
        alpha=0.35,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude (°E)", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Latitude (°N)", color="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = ax.set_title("", color="#f0f6fc", fontsize=13, pad=14, loc="left")
    fig.text(
        0.08, 0.03,
        f"Mask: elevation ≥ {elev_min:.0f} m  ·  lowlands hidden",
        color="#8b949e",
        fontsize=10,
    )

    def update(frame: int):
        im_cloud.set_data(cloud_rgba(field[frame], mask, vmin, vmax))
        ts = wall_clock(time_var, frame) or f"timestep {frame}"
        title.set_text(f"Bavarian Alps — cloud cover\n{ts}   ·   frame {frame + 1}/{n_time}")
        return im_cloud, title

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
    parser.add_argument("--var", default="clt")
    parser.add_argument(
        "--output",
        type=Path,
        default=CLOUDS_ROOT / "output" / "movies" / "alps_irregular_cropped.gif",
    )
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument(
        "--elev-min",
        type=float,
        default=DEFAULT_ELEV_MIN,
        help="Only show terrain at or above this elevation (m)",
    )
    parser.add_argument("--vert-exag", type=float, default=1.8, help="Hillshade vertical exaggeration")
    parser.add_argument("--pad-deg", type=float, default=0.04, help="Padding around mask bbox (degrees)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=130)
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
        args.elev_min,
        args.vert_exag,
        args.pad_deg,
        args.fps,
        args.dpi,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
