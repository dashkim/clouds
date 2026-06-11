#!/usr/bin/env python3
"""Write a markdown inventory of a 2d_lonlat*.nc file."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import CLOUDS_ROOT, add_input_arg, find_lonlat_nc

CLOUD_KEYWORDS = (
    "clw", "cli", "clt", "cct", "ccb", "qc", "qr", "hus", "cloud", "ice", "water", "precip",
)
GRID_NAMES = {"lon", "lat", "height", "height_2", "time"}


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


def _parse_time_base(units: str) -> datetime | None:
    match = re.match(r"days since (.+)", units)
    if not match:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(match.group(1), fmt)
        except ValueError:
            continue
    return None


def _time_summary(ds: nc.Dataset) -> list[str]:
    if "time" not in ds.variables:
        return ["No `time` variable."]
    t = ds.variables["time"]
    n = len(t)
    units = getattr(t, "units", "?")
    lines = [
        f"- Length: **{n}** timestep(s)",
        f"- `units`: {units}",
        f"- `calendar`: {getattr(t, 'calendar', 'standard')}",
    ]
    if not n:
        return lines

    vals = t[:]
    lines.append(f"- First value: `{vals[0]}`")
    if n > 1:
        lines.append(f"- Last value: `{vals[-1]}`")
        dt_sec = float((vals[1] - vals[0]) * 86400)
        lines.append(f"- Step interval: **{dt_sec:.0f} s**")

    base = _parse_time_base(units)
    if base is not None:
        first = base + timedelta(days=float(vals[0]))
        last = base + timedelta(days=float(vals[-1]))
        span = last - first
        lines.append(f"- Wall-clock span: **{first:%Y-%m-%d %H:%M:%S}** → **{last:%H:%M:%S}** ({span})")
    return lines


def _horizontal_summary(ds: nc.Dataset) -> list[str]:
    lines: list[str] = []
    if "lon" in ds.variables:
        lon = ds.variables["lon"][:]
        lines.append(
            f"- **`lon`**: {len(lon)} points, units `{getattr(ds.variables['lon'], 'units', '?')}`"
        )
        lines.append(f"  - Range: {float(lon.min()):.4f} … {float(lon.max()):.4f} °E")
    if "lat" in ds.variables:
        lat = ds.variables["lat"][:]
        lines.append(
            f"- **`lat`**: {len(lat)} points, units `{getattr(ds.variables['lat'], 'units', '?')}`"
        )
        lines.append(f"  - Range: {float(lat.min()):.4f} … {float(lat.max()):.4f} °N")
    return lines or ["No lon/lat coordinate variables found."]


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
        n_lon = len(ds.dimensions["lon"]) if "lon" in ds.dimensions else 0
        n_lat = len(ds.dimensions["lat"]) if "lat" in ds.dimensions else 0
        lines = [
            "# LON/LAT 2D NetCDF inventory",
            "",
            f"- **File:** `{nc_path.name}`",
            f"- **Path:** `{nc_path}`",
            f"- **Size:** {size_gb:.2f} GiB",
            f"- **Generated:** {now}",
            "",
            "## Format summary",
            "",
            (
                f"Regular lon/lat grid ({n_lon} × {n_lat}); "
                f"surface and cloud-diagnostic fields on a 10 s time series."
            ),
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
        lines.extend(["", "## Horizontal coordinates", "", *_horizontal_summary(ds)])
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
            f"- Timesteps: **{n_time}** at 10 s resolution — animate `clt` or other 2D fields over time.",
            f"- Grid: **{n_lon} × {n_lat}** — suitable for map-style 2D plots (matplotlib, ParaView, VisIt).",
            "- Static terrain: `z_ifc` is time-independent (height above sea level).",
            "- Metadata quirk: `cct` long_name says cloud top pressure but `units` is `K` in this file; verify before plotting.",
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
    add_input_arg(parser, kind="lonlat")
    parser.add_argument(
        "-o", "--output", type=Path, default=CLOUDS_ROOT / "reports" / "inventories" / "lonlat_report.md"
    )
    args = parser.parse_args()
    nc_path = find_lonlat_nc(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(nc_path), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
