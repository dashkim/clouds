#!/usr/bin/env python3
"""Phase 1: explore inversion-like conditions in local NetCDF data."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.netcdf import fill_value, mask_invalid, wall_clock
from lib.paths import CLOUDS_ROOT, add_input_arg, find_icon_dom_nc, find_lonlat_nc

# Expected dimensions from inventory reports
EXP_LONLAT_TIME = 121
EXP_LON = 1429
EXP_LAT = 1556
EXP_ICON_TIME = 1
EXP_HEIGHT = 150

CLT_HIGH_MAX = 25.0
CLT_LOW_MIN = 50.0
CLT_PIXEL_MAX = 20.0
ALPINE_LAT_MIN = 47.8

ALPS_LON_MIN, ALPS_LON_MAX = 10.0, 13.0
ALPS_LAT_MIN, ALPS_LAT_MAX = 47.0, 48.0

CLOUD_MIX_THRESHOLD = 1e-6
INVERSION_MIN_DEPTH_M = 200.0
PROFILE_SAMPLE_COUNT = 20


def smoke_test_lonlat(path: Path) -> dict:
    with nc.Dataset(path, "r") as ds:
        n_time = len(ds.dimensions["time"])
        n_lon = len(ds.dimensions["lon"])
        n_lat = len(ds.dimensions["lat"])
        required = ("clt", "z_ifc", "lon", "lat", "time")
        missing = [v for v in required if v not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables in {path.name}: {missing}")
        if n_time != EXP_LONLAT_TIME or n_lon != EXP_LON or n_lat != EXP_LAT:
            print(
                f"Warning: lon/lat dims {n_time}/{n_lat}/{n_lon} differ from "
                f"expected {EXP_LONLAT_TIME}/{EXP_LAT}/{EXP_LON}",
                file=sys.stderr,
            )
        t0 = wall_clock(ds.variables["time"], 0)
        t1 = wall_clock(ds.variables["time"], n_time - 1)
        return {
            "path": str(path),
            "n_time": n_time,
            "n_lon": n_lon,
            "n_lat": n_lat,
            "time_start": t0,
            "time_end": t1,
        }


def smoke_test_icon(path: Path) -> dict:
    with nc.Dataset(path, "r") as ds:
        n_time = len(ds.dimensions["time"])
        n_height = len(ds.dimensions["height"])
        n_cells = len(ds.dimensions["ncells"])
        required = ("clw", "cli", "ta", "height", "clon", "clat")
        missing = [v for v in required if v not in ds.variables]
        if missing:
            raise KeyError(f"Missing variables in {path.name}: {missing}")
        if n_time != EXP_ICON_TIME:
            print(f"Warning: icon time={n_time}, expected {EXP_ICON_TIME}", file=sys.stderr)
        if n_height != EXP_HEIGHT:
            print(f"Warning: icon height={n_height}, expected {EXP_HEIGHT}", file=sys.stderr)
        t0 = wall_clock(ds.variables.get("time"), 0) if "time" in ds.variables else None
        return {
            "path": str(path),
            "n_time": n_time,
            "n_height": n_height,
            "n_cells": n_cells,
            "time": t0,
        }


def load_terrain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(path, "r") as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        z_var = ds.variables["z_ifc"]
        z = mask_invalid(np.asarray(z_var[:], dtype=np.float32), fill_value(z_var))
    return lon, lat, z


def scan_2d(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
) -> dict:
    valid_z = z[np.isfinite(z)]
    z_high_thresh = float(np.nanpercentile(valid_z, 75))
    z_low_thresh = float(np.nanpercentile(valid_z, 25))

    high_mask = np.isfinite(z) & (z >= z_high_thresh)
    low_mask = np.isfinite(z) & (z <= z_low_thresh)
    alpine_mask = high_mask & (lat[:, None] >= ALPINE_LAT_MIN)
    non_alpine_high = high_mask & ~alpine_mask

    lat_2d = lat[:, None]
    n_time = EXP_LONLAT_TIME

    timestep_like: list[bool] = []
    mean_clt_high: list[float] = []
    mean_clt_low: list[float] = []
    candidate_fraction: list[float] = []
    alpine_candidate_fraction: list[float] = []

    max_t = 0
    min_t = 0
    max_frac = -1.0
    min_frac = 2.0
    max_clt_snapshot: np.ndarray | None = None
    min_clt_snapshot: np.ndarray | None = None

    with nc.Dataset(path, "r") as ds:
        clt_var = ds.variables["clt"]
        fill = fill_value(clt_var)

        for t in range(n_time):
            clt = mask_invalid(np.asarray(clt_var[t], dtype=np.float32), fill)

            clt_high = float(np.nanmean(clt[high_mask])) if np.any(high_mask) else np.nan
            clt_low = float(np.nanmean(clt[low_mask])) if np.any(low_mask) else np.nan
            mean_clt_high.append(clt_high)
            mean_clt_low.append(clt_low)

            like = (
                np.isfinite(clt_high)
                and np.isfinite(clt_low)
                and clt_high < CLT_HIGH_MAX
                and clt_low > CLT_LOW_MIN
            )
            timestep_like.append(like)

            low_clt_neighborhood = np.nanmean(clt[low_mask]) if np.any(low_mask) else np.nan
            candidates = (
                high_mask
                & np.isfinite(clt)
                & (clt < CLT_PIXEL_MAX)
                & np.isfinite(low_clt_neighborhood)
                & (low_clt_neighborhood > CLT_LOW_MIN)
            )
            frac = float(np.sum(candidates) / np.sum(high_mask)) if np.any(high_mask) else 0.0
            candidate_fraction.append(frac)

            if np.any(alpine_mask):
                alp_frac = float(np.sum(candidates & alpine_mask) / np.sum(alpine_mask))
            else:
                alp_frac = 0.0
            alpine_candidate_fraction.append(alp_frac)

            if frac > max_frac:
                max_frac = frac
                max_t = t
                max_clt_snapshot = clt.copy()
            if frac < min_frac:
                min_frac = frac
                min_t = t
                min_clt_snapshot = clt.copy()

    n_like = sum(timestep_like)
    pct_like = 100.0 * n_like / n_time

    return {
        "z_high_thresh": z_high_thresh,
        "z_low_thresh": z_low_thresh,
        "timestep_like": timestep_like,
        "mean_clt_high": mean_clt_high,
        "mean_clt_low": mean_clt_low,
        "candidate_fraction": candidate_fraction,
        "alpine_candidate_fraction": alpine_candidate_fraction,
        "n_like": n_like,
        "pct_like": pct_like,
        "max_t": max_t,
        "min_t": min_t,
        "max_clt_snapshot": max_clt_snapshot,
        "min_clt_snapshot": min_clt_snapshot,
        "high_mask": high_mask,
        "low_mask": low_mask,
        "alpine_mask": alpine_mask,
        "non_alpine_high": non_alpine_high,
        "lat_2d": lat_2d,
    }


def _find_inversion_top(height: np.ndarray, ta: np.ndarray) -> float | None:
    """Return height (m) at top of lowest qualifying inversion layer, or None."""
    if len(height) < 2:
        return None
    dh = np.diff(height)
    dt = np.diff(ta)
    valid = np.isfinite(dh) & (dh > 0) & np.isfinite(dt)
    if not np.any(valid):
        return None

    inv = valid & (dt / dh > 0)
    i = 0
    while i < len(inv):
        if not inv[i]:
            i += 1
            continue
        start = i
        depth = 0.0
        while i < len(inv) and inv[i]:
            depth += float(dh[i])
            i += 1
        if depth >= INVERSION_MIN_DEPTH_M:
            return float(height[i])
    return None


def _cloud_below_clear_above(
    height: np.ndarray,
    mix: np.ndarray,
    inv_top: float,
) -> bool:
    below = height < inv_top
    above = height >= inv_top
    if not np.any(below) or not np.any(above):
        return False
    below_cloudy = float(np.nanmax(mix[below])) > CLOUD_MIX_THRESHOLD
    above_clear = float(np.nanmax(mix[above])) < CLOUD_MIX_THRESHOLD
    return below_cloudy and above_clear


def scan_3d(path: Path, fig_dir: Path) -> dict:
    with nc.Dataset(path, "r") as ds:
        clon = np.rad2deg(np.asarray(ds.variables["clon"][:], dtype=np.float64))
        clat = np.rad2deg(np.asarray(ds.variables["clat"][:], dtype=np.float64))
        height = np.asarray(ds.variables["height"][:], dtype=np.float64)
        # Variables are (height, ncells)
        ta_all = np.asarray(ds.variables["ta"][0], dtype=np.float32)
        pres_all = np.asarray(ds.variables["pres"][0], dtype=np.float32)
        clw = np.asarray(ds.variables["clw"][0], dtype=np.float32)
        cli = np.asarray(ds.variables["cli"][0], dtype=np.float32)
        mix = clw + cli

    alps = (
        (clon >= ALPS_LON_MIN)
        & (clon <= ALPS_LON_MAX)
        & (clat >= ALPS_LAT_MIN)
        & (clat <= ALPS_LAT_MAX)
    )
    # Lower surface pressure ≈ higher elevation (lowest height index)
    surface_pres = pres_all[0, :]
    finite = np.isfinite(surface_pres)
    if np.sum(finite & alps) < PROFILE_SAMPLE_COUNT:
        alps = finite

    alps_idx = np.where(alps)[0]
    if alps_idx.size == 0:
        return {"n_sampled": 0, "n_inversion": 0, "n_cloud_structure": 0, "profile_paths": []}

    elev_thresh = float(np.nanpercentile(surface_pres[alps_idx], 25))
    candidates = alps_idx[surface_pres[alps_idx] <= elev_thresh]
    if candidates.size > PROFILE_SAMPLE_COUNT:
        rng = np.random.default_rng(0)
        candidates = rng.choice(candidates, size=PROFILE_SAMPLE_COUNT, replace=False)

    n_inversion = 0
    n_cloud_structure = 0
    profile_paths: list[str] = []

    for i, cell in enumerate(candidates):
        ta_prof = ta_all[:, cell]
        mix_prof = mix[:, cell]
        inv_top = _find_inversion_top(height, ta_prof)
        has_inv = inv_top is not None
        has_cloud = False
        if has_inv:
            n_inversion += 1
            has_cloud = _cloud_below_clear_above(height, mix_prof, inv_top)
            if has_cloud:
                n_cloud_structure += 1

        if i < 6 and (has_inv or i < 3):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(ta_prof, height, "r-", lw=1.5)
            axes[0].set_xlabel("Temperature (K)")
            axes[0].set_ylabel("Height (m)")
            axes[0].set_title(f"Cell {cell}: lon={clon[cell]:.2f}° lat={clat[cell]:.2f}°")
            axes[0].grid(True, alpha=0.3)
            if inv_top is not None:
                axes[0].axhline(inv_top, color="gray", ls="--", label="inversion top")
                axes[0].legend()

            axes[1].semilogy(mix_prof + 1e-12, height, "b-", lw=1.5)
            axes[1].set_xlabel("CLW + CLI (kg/kg)")
            axes[1].set_ylabel("Height (m)")
            axes[1].set_title("Cloud mixing ratio")
            axes[1].grid(True, alpha=0.3)
            if inv_top is not None:
                axes[1].axhline(inv_top, color="gray", ls="--")

            fig.tight_layout()
            out = fig_dir / f"profile_cell_{cell}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            profile_paths.append(str(out.relative_to(CLOUDS_ROOT)))

    return {
        "n_sampled": len(candidates),
        "n_inversion": n_inversion,
        "n_cloud_structure": n_cloud_structure,
        "profile_paths": profile_paths,
    }


def plot_2d_figures(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    scan: dict,
    fig_dir: Path,
) -> list[str]:
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scan["candidate_fraction"], "b-", label="All high terrain")
    ax.plot(scan["alpine_candidate_fraction"], "g-", label="Alpine high terrain")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Fraction of high-terrain pixels (inversion candidates)")
    ax.set_title("Inversion candidate fraction over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "candidate_fraction_timeseries.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p.relative_to(CLOUDS_ROOT)))

    for label, t_idx, clt_snap in (
        ("max_signal", scan["max_t"], scan["max_clt_snapshot"]),
        ("min_signal", scan["min_t"], scan["min_clt_snapshot"]),
    ):
        if clt_snap is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].pcolormesh(lon, lat, z, shading="auto", cmap="terrain")
        plt.colorbar(im0, ax=axes[0], label="z_ifc (m)")
        axes[0].set_title("Terrain elevation")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        im1 = axes[1].pcolormesh(lon, lat, clt_snap, shading="auto", cmap="Blues", vmin=0, vmax=100)
        plt.colorbar(im1, ax=axes[1], label="clt (%)")
        axes[1].set_title(f"Cloud cover — timestep {t_idx} ({label})")
        axes[1].set_xlabel("Longitude")

        fig.tight_layout()
        p = fig_dir / f"clt_{label}_t{t_idx}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p.relative_to(CLOUDS_ROOT)))

    return paths


def decision_gate(scan2d: dict, scan3d: dict) -> tuple[str, str]:
    pct = scan2d["pct_like"]
    n_inv = scan3d["n_inversion"]
    n_cloud = scan3d["n_cloud_structure"]

    if pct >= 10.0 or n_cloud >= 3:
        return (
            "Go",
            "Proceed to Phase 2 on local data. Inversion-like structure is present. "
            "Optionally download longer WDCC sequences later for ML training.",
        )
    if pct >= 5.0 or n_inv >= 5:
        return (
            "Go (weak)",
            "Marginal signal detected. Phase 2 is viable on local data, but downloading "
            "`2d_6hours.nc` from WDCC is recommended before ML.",
        )
    if pct > 0 or n_inv > 0:
        return (
            "Download",
            "Weak/rare inversion signal in the 20-minute window. Download longer 2D "
            "sequences (and multi-timestep 3D if available) from WDCC before Phase 2/ML.",
        )
    return (
        "Pivot",
        "No inversion-like signal detected. Proceed with baseline cloud movie only; "
        "revisit inversion definition or select a different time window from WDCC.",
    )


def write_findings(
    out_path: Path,
    lonlat_info: dict,
    icon_info: dict,
    scan2d: dict,
    scan3d: dict,
    fig_paths: list[str],
    verdict: str,
    recommendation: str,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Phase 1 findings — inversion exploration",
        "",
        f"- **Generated:** {now}",
        f"- **2D file:** `{lonlat_info['path']}`",
        f"- **3D file:** `{icon_info['path']}`",
        "",
        "## Smoke test",
        "",
        f"- 2D grid: {scan2d.get('n_lon', lonlat_info['n_lon'])} × {scan2d.get('n_lat', lonlat_info['n_lat'])}, "
        f"{lonlat_info['n_time']} timesteps",
        f"- 2D time span: {lonlat_info['time_start']} → {lonlat_info['time_end']}",
        f"- 3D grid: {icon_info['n_cells']:,} cells, {icon_info['n_height']} height levels, "
        f"{icon_info['n_time']} timestep(s)",
        f"- 3D time: {icon_info['time'] or 'n/a'}",
        "",
        "## 2D summit-clear / valley-cloudy scan",
        "",
        f"- High terrain threshold (75th pct `z_ifc`): **{scan2d['z_high_thresh']:.0f} m**",
        f"- Low terrain threshold (25th pct `z_ifc`): **{scan2d['z_low_thresh']:.0f} m**",
        f"- Inversion-like timesteps (mean clt high < {CLT_HIGH_MAX}% AND low > {CLT_LOW_MIN}%): "
        f"**{scan2d['n_like']} / {lonlat_info['n_time']}** ({scan2d['pct_like']:.1f}%)",
        f"- Max candidate fraction at timestep **{scan2d['max_t']}**; "
        f"min at timestep **{scan2d['min_t']}**",
        "",
        "## 3D vertical profile sampling (Alps region)",
        "",
        f"- Cells sampled: **{scan3d['n_sampled']}**",
        f"- Temperature inversion layers (≥ {INVERSION_MIN_DEPTH_M:.0f} m): **{scan3d['n_inversion']}**",
        f"- Inversion + cloud-below-clear-above: **{scan3d['n_cloud_structure']}**",
        "",
        "**Note:** 3D ICON grid and 2D lon/lat grid are not co-registered; analyses are independent.",
        "",
        "## Figures",
        "",
    ]
    for p in fig_paths + scan3d.get("profile_paths", []):
        lines.append(f"- `{p}`")
    lines.extend([
        "",
        "## Decision gate",
        "",
        f"**Verdict: {verdict}**",
        "",
        recommendation,
        "",
        "## WDCC download decision",
        "",
    ])
    if verdict == "Pivot":
        lines.append(
            "Download a longer or different time window from WDCC (e.g. `2d_6hours.nc`) and "
            "revisit inversion proxies (consider fixed elevation thresholds for Alps, e.g. "
            "≥ 1500 m) before Phase 2 or ML."
        )
    elif verdict in ("Download", "Go (weak)"):
        lines.append(
            "Download `2d_6hours.nc` (and multi-timestep 3D domains if needed) from "
            "[WDCC SciVis 2017](https://scivis2017.dkrz.de/) before investing in ML."
        )
    else:
        lines.append(
            "No immediate WDCC download required for Phase 2. Revisit if ML training needs "
            "more timesteps."
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    parser.add_argument("--icon-input", help="Path to 3d_icon_dom*.nc")
    parser.add_argument(
        "--report",
        type=Path,
        default=CLOUDS_ROOT / "reports" / "phase01" / "findings.md",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=CLOUDS_ROOT / "output" / "phase01",
    )
    args = parser.parse_args()

    lonlat_path = find_lonlat_nc(args.input)
    icon_path = find_icon_dom_nc(args.icon_input)

    print("Smoke test...")
    lonlat_info = smoke_test_lonlat(lonlat_path)
    icon_info = smoke_test_icon(icon_path)
    print(f"  2D: {lonlat_info['n_time']} steps, {lonlat_info['time_start']} → {lonlat_info['time_end']}")
    print(f"  3D: {icon_info['n_cells']:,} cells, {icon_info['n_height']} levels")

    print("Loading terrain...")
    lon, lat, z = load_terrain(lonlat_path)

    print("2D inversion scan...")
    scan2d = scan_2d(lonlat_path, lon, lat, z)
    scan2d["n_lon"] = lonlat_info["n_lon"]
    scan2d["n_lat"] = lonlat_info["n_lat"]
    print(f"  Inversion-like timesteps: {scan2d['n_like']}/{lonlat_info['n_time']} ({scan2d['pct_like']:.1f}%)")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    print("Plotting 2D figures...")
    fig_paths = plot_2d_figures(lon, lat, z, scan2d, args.fig_dir)

    print("3D profile sampling...")
    scan3d = scan_3d(icon_path, args.fig_dir)
    print(f"  Inversions: {scan3d['n_inversion']}, cloud structure: {scan3d['n_cloud_structure']}")

    verdict, recommendation = decision_gate(scan2d, scan3d)
    print(f"Decision gate: {verdict}")

    write_findings(
        args.report,
        lonlat_info,
        icon_info,
        scan2d,
        scan3d,
        fig_paths,
        verdict,
        recommendation,
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
