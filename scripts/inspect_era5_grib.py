#!/usr/bin/env python3
"""Inventory ERA5 GRIB groups, variables, and time coverage."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.era5_io import inventory_grib
from lib.paths import CLOUDS_ROOT, add_era5_arg, find_era5_grib


def format_inventory_md(rows: list[dict], grib_path: Path) -> str:
    lines = [
        "# ERA5 GRIB inventory",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **File:** `{grib_path}`",
        f"- **Size:** {grib_path.stat().st_size / (1024 * 1024):.1f} MiB",
        f"- **Groups:** {len(rows)}",
        "",
        "## Unit notes",
        "",
        "- ERA5 `tcc`, `lcc`, `mcc`, `hcc` are **0–1**; ICON `clt` is **0–100 %**.",
        "- `cbh` is in **meters**; ICON `ccb` is pressure (Pa) → convert with `pressure_to_height_msl`.",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## Group {row['group']}",
            "",
            f"- **Sizes:** `{row['sizes']}`",
            f"- **Time:** {row['time_start']} → {row['time_end']} ({row['n_time']} steps)",
            f"- **Grid spacing:** Δlat={row['dlat']}°, Δlon={row['dlon']}°",
            "",
            "| Variable | shortName | units | dims |",
            "|----------|-----------|-------|------|",
        ])
        for v in row["variables"]:
            lines.append(
                f"| {v['name']} | {v['shortName']} | {v['units']} | {', '.join(v['dims'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_era5_arg(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=CLOUDS_ROOT / "reports" / "phase02" / "era5_inventory.md",
    )
    args = parser.parse_args()

    grib_path = find_era5_grib(args.era5_grib)
    rows = inventory_grib(grib_path)
    report = format_inventory_md(rows, grib_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"Group {row['group']}: {row['n_time']} times, "
            f"{len(row['variables'])} variables"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
