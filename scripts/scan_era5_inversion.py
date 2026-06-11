#!/usr/bin/env python3
"""Scan ERA5 hourly timesteps for inversion fraction per patch region."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.alps_region import resolve_patch_bounds
from lib.era5_io import load_era5_hourly_fields
from lib.inversion import (
    InversionParams,
    cover_to_percent,
    detect_inversion_mask,
    inversion_fraction,
    ridge_above_deck_fraction,
    ridge_inversion_fraction,
)
from lib.paths import CLOUDS_ROOT, add_era5_arg, find_era5_grib, find_lonlat_nc
from lib.patch_fields import _load_icon_terrain
from lib.alps_region import crop_indices, default_patch_stride

import netCDF4 as nc


def scan_region(
    *,
    region: str,
    grib_path: Path,
    icon_path: Path,
    inversion_params: InversionParams,
    stride: int,
    max_hours: int | None,
) -> list[dict]:
    z_min_m = inversion_params.z_min_m
    lon_min, lon_max, lat_min, lat_max, _, _ = resolve_patch_bounds(region=region)

    with nc.Dataset(icon_path, "r") as ds:
        lon_sl, lat_sl = crop_indices(
            ds.variables["lon"][:],
            ds.variables["lat"][:],
            lon_min, lon_max, lat_min, lat_max,
        )
    lon, lat, z = _load_icon_terrain(icon_path, lon_sl, lat_sl, stride)

    from lib.era5_io import cbh_at_index, era5_to_icon_grid, preload_cbh_series, step_dataset

    ds_hourly = load_era5_hourly_fields(
        grib_path,
        lon_min=lon_min - 0.5,
        lon_max=lon_max + 0.5,
        lat_min=lat_min - 0.5,
        lat_max=lat_max + 0.5,
    )
    preload_cbh_series(grib_path)
    times = pd.to_datetime(ds_hourly["time"].values)
    n_time = len(times)
    if max_hours is not None:
        n_time = min(n_time, max_hours)

    era5_lat = np.asarray(ds_hourly["latitude"].values, dtype=np.float64)
    era5_lon = np.asarray(ds_hourly["longitude"].values, dtype=np.float64)
    tcc_all = np.asarray(ds_hourly["tcc"].values, dtype=np.float32)
    lcc_all = np.asarray(ds_hourly["lcc"].values, dtype=np.float32)
    mcc_all = np.asarray(ds_hourly["mcc"].values, dtype=np.float32)
    ds_step = step_dataset(grib_path)
    cbh_lat = np.asarray(ds_step["latitude"].values, dtype=np.float64)
    cbh_lon = np.asarray(ds_step["longitude"].values, dtype=np.float64)

    rows: list[dict] = []
    for t in range(n_time):
        tcc = era5_to_icon_grid(tcc_all[t], era5_lat, era5_lon, lat, lon)
        lcc = era5_to_icon_grid(lcc_all[t], era5_lat, era5_lon, lat, lon)
        mcc = era5_to_icon_grid(mcc_all[t], era5_lat, era5_lon, lat, lon)
        cbh = era5_to_icon_grid(
            cbh_at_index(grib_path, t, times.to_numpy()), cbh_lat, cbh_lon, lat, lon,
        )

        mask = detect_inversion_mask(
            z,
            cloud_cover=tcc,
            low_cover=lcc,
            medium_cover=mcc,
            cloud_base_m=cbh,
            params=inversion_params,
        )
        inv_frac = inversion_fraction(mask)
        ridge_frac = ridge_inversion_fraction(mask, z, z_min_m=z_min_m)
        deck_frac = ridge_above_deck_fraction(
            z, cbh, z_min_m=z_min_m, margin_m=inversion_params.cbh_margin_m,
            cloud_cover=tcc,
        )
        rows.append({
            "region": region,
            "inversion_mode": inversion_params.mode,
            "time": str(times[t]),
            "inversion_fraction": inv_frac,
            "ridge_inversion_fraction": ridge_frac,
            "ridge_above_deck_fraction": deck_frac,
            "mean_tcc_pct": float(np.nanmean(cover_to_percent(tcc))),
            "mean_lcc_pct": float(np.nanmean(cover_to_percent(lcc))),
        })
        if (t + 1) % 240 == 0:
            print(f"  {region}: {t + 1}/{n_time} hours scanned")
    return rows


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(CLOUDS_ROOT))
    except ValueError:
        return str(path)


def write_report(all_rows: list[dict], out_md: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# ERA5 inversion timeline",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **CSV:** `{_display_path(out_csv)}`",
        "",
    ]
    for region in sorted({r["region"] for r in all_rows}):
        modes = sorted({r["inversion_mode"] for r in all_rows if r["region"] == region})
        for mode in modes:
            region_rows = [
                r for r in all_rows if r["region"] == region and r["inversion_mode"] == mode
            ]
            ranked = sorted(
                region_rows, key=lambda r: r["ridge_inversion_fraction"], reverse=True,
            )
            lines.extend([
                f"## {region} ({mode})",
                "",
                f"- Hours scanned: **{len(region_rows)}**",
                f"- Max ridge inversion fraction: **{ranked[0]['ridge_inversion_fraction']:.1%}** "
                f"at `{ranked[0]['time']}`",
                "",
                "### Top 20 hours (ridge inversion fraction)",
                "",
                "| time | ridge inv % | ridge above deck % | grid inv % | mean lcc % |",
                "|------|-------------|----------------------|------------|------------|",
            ])
            for r in ranked[:20]:
                lines.append(
                    f"| {r['time']} | {r['ridge_inversion_fraction']:.1%} | "
                    f"{r['ridge_above_deck_fraction']:.1%} | "
                    f"{r['inversion_fraction']:.1%} | {r['mean_lcc_pct']:.1f} |"
                )
            lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_era5_arg(parser)
    parser.add_argument(
        "--region",
        choices=("west", "east", "east_core", "all"),
        default="east_core",
    )
    parser.add_argument("--z-min-m", type=float, default=800.0)
    parser.add_argument(
        "--inversion-mode",
        choices=("deck_only", "phenomenological"),
        default="phenomenological",
    )
    parser.add_argument("--cbh-margin-m", type=float, default=100.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-hours", type=int, default=None, help="Limit scan for quick tests")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=CLOUDS_ROOT / "reports" / "phase02" / "inversion_timeline.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CLOUDS_ROOT / "output" / "phase02" / "inversion_fraction_by_hour.csv",
    )
    args = parser.parse_args()

    grib_path = find_era5_grib(args.era5_grib)
    icon_path = find_lonlat_nc()
    regions = ("west", "east", "east_core") if args.region == "all" else (args.region,)

    inversion_params = InversionParams(
        mode=args.inversion_mode,
        z_min_m=args.z_min_m,
        cbh_margin_m=args.cbh_margin_m,
    )

    all_rows: list[dict] = []
    for region in regions:
        stride = args.stride if args.stride is not None else default_patch_stride(region)
        print(f"Scanning {region} (stride={stride}, mode={args.inversion_mode})...")
        all_rows.extend(scan_region(
            region=region,
            grib_path=grib_path,
            icon_path=icon_path,
            inversion_params=inversion_params,
            stride=stride,
            max_hours=args.max_hours,
        ))

    write_report(all_rows, args.output_md, args.output_csv)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
