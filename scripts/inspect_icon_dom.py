#!/usr/bin/env python3
"""Write a markdown inventory of a 3d_icon_dom*.nc file."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import CLOUDS_ROOT, add_input_arg, find_icon_dom_nc

CLOUD_KEYWORDS = ("clw", "cli", "qc", "qr", "hus", "cloud", "ice", "water", "precip")
GRID_NAMES = {
    "clon", "clat", "clon_bnds", "clat_bnds", "clon_vertices", "clat_vertices",
    "lon", "lat", "height", "height_2", "time",
}


def _attr_lines(obj) -> list[str]:
    lines: list[str] = []
    for key in sorted(obj.ncattrs()):
        val = getattr(obj, key)
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        text = str(val)
        if len(text) > 200:
            text = text[:200] + "…"
        lines.append(f"  - `{key}`: {text}")
    return lines


def _is_cloud_candidate(name: str, var) -> bool:
    lower = name.lower()
    if any(k in lower for k in CLOUD_KEYWORDS):
        return True
    for attr in ("standard_name", "long_name"):
        if attr in var.ncattrs():
            text = str(getattr(var, attr)).lower()
            if any(k in text for k in CLOUD_KEYWORDS):
                return True
    return False


def _time_summary(ds: nc.Dataset) -> list[str]:
    if "time" not in ds.variables:
        return ["No `time` variable."]
    t = ds.variables["time"]
    n = len(t)
    lines = [
        f"- Length: **{n}** timestep(s)",
        f"- `units`: {getattr(t, 'units', '?')}",
        f"- `calendar`: {getattr(t, 'calendar', 'standard')}",
    ]
    if n:
        vals = t[:]
        lines.append(f"- First value: `{vals[0]}`")
        if n > 1:
            lines.append(f"- Last value: `{vals[-1]}`")
    return lines


def _vertical_summary(ds: nc.Dataset) -> list[str]:
    lines: list[str] = []
    for name in ("height", "height_2", "level", "lev"):
        if name not in ds.variables:
            continue
        z = ds.variables[name]
        lines.append(f"- **`{name}`**: {len(z)} levels, units `{getattr(z, 'units', '?')}`")
        if len(z):
            zvals = z[:]
            lines.append(f"  - Range: {float(zvals.min()):.2f} … {float(zvals.max()):.2f}")
    return lines or ["No vertical coordinate variables found."]


def build_report(nc_path: Path) -> str:
    size_gb = nc_path.stat().st_size / (1024**3)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with nc.Dataset(nc_path, "r") as ds:
        n_time = len(ds.dimensions["time"]) if "time" in ds.dimensions else 0
        lines = [
            "# ICON DOM NetCDF inventory",
            "",
            f"- **File:** `{nc_path.name}`",
            f"- **Path:** `{nc_path}`",
            f"- **Size:** {size_gb:.2f} GiB",
            f"- **Generated:** {now}",
            "",
            "## Format summary",
            "",
            "ICON unstructured triangular grid; see dimensions and variables below.",
            "",
            "## Global attributes",
            "",
            *(_attr_lines(ds) or ["  (none)"]),
            "",
            "## Dimensions",
            "",
            "| Name | Length |",
            "|------|--------|",
        ]
        for name, dim in ds.dimensions.items():
            length = len(dim) if not dim.isunlimited() else f"{len(dim)} (unlimited)"
            lines.append(f"| `{name}` | {length} |")
        lines.extend(["", "## Time axis", "", *_time_summary(ds)])
        lines.extend(["", "## Vertical coordinates", "", *_vertical_summary(ds)])
        lines.extend([
            "",
            "## Grid / coordinate variables",
            "",
            "| Variable | Shape | dtype | Notes |",
            "|----------|-------|-------|-------|",
        ])
        for name, var in ds.variables.items():
            if name in GRID_NAMES:
                lines.append(
                    f"| `{name}` | `{var.shape}` | {var.dtype} | units: {getattr(var, 'units', '')} |"
                )
        lines.extend([
            "",
            "## Data variables",
            "",
            "| Variable | Dimensions | dtype | long_name | units |",
            "|----------|------------|-------|-----------|-------|",
        ])
        cloud_vars = []
        for name, var in ds.variables.items():
            if name in GRID_NAMES or not var.dimensions:
                continue
            lines.append(
                f"| `{name}` | {var.dimensions} | {var.dtype} | "
                f"{getattr(var, 'long_name', '')} | {getattr(var, 'units', '')} |"
            )
            if _is_cloud_candidate(name, var):
                cloud_vars.append(name)
        lines.extend(["", "## Cloud-related variables", ""])
        if cloud_vars:
            for name in cloud_vars:
                var = ds.variables[name]
                lines.append(f"- **`{name}`**: dims {var.dimensions}")
        else:
            lines.append("- (none auto-detected)")
        lines.extend([
            "",
            "## Largest arrays",
            "",
            "| Variable | Elements |",
            "|----------|----------|",
        ])
        for name, var in ds.variables.items():
            if var.dimensions and var.size > 1_000_000:
                lines.append(f"| `{name}` | {var.size:,} |")
        lines.extend([
            "",
            "## Visualization notes",
            "",
            f"- Timesteps: **{n_time}**. "
            + ("Use `--mode time` in the VTK viewer." if n_time > 1 else "Use `--mode height` to sweep vertical levels."),
            "- Subsample `ncells` for interactive VTK.",
            "",
        ])
    try:
        header = subprocess.run(
            ["ncdump", "-h", str(nc_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if header.stdout:
            lines.extend(["## Appendix: ncdump -h", "", "```", header.stdout.strip(), "```", ""])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser)
    parser.add_argument(
        "-o", "--output", type=Path, default=CLOUDS_ROOT / "reports" / "inventories" / "icon_dom_report.md"
    )
    args = parser.parse_args()
    nc_path = find_icon_dom_nc(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(nc_path), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
