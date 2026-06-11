"""Shared NetCDF helpers."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np

FILL_THRESHOLD = 1e30


def parse_time_base(units: str) -> datetime | None:
    match = re.match(r"days since (.+)", units)
    if not match:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(match.group(1), fmt)
        except ValueError:
            continue
    return None


def wall_clock(
    time_var,
    time_idx: int,
    *,
    time_units: str | None = None,
    time_values: np.ndarray | None = None,
) -> str | None:
    if time_values is not None and time_units is not None:
        units = time_units
        days = float(time_values[time_idx])
    elif time_var is not None:
        units = getattr(time_var, "units", "")
        try:
            days = float(time_var[time_idx])
        except (RuntimeError, IndexError, TypeError):
            return None
    else:
        return None
    base = parse_time_base(units)
    if base is None:
        return None
    return (base + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def fill_value(var) -> float | None:
    for attr in ("_FillValue", "missing_value"):
        if attr in var.ncattrs():
            return float(getattr(var, attr))
    return None


def mask_invalid(values: np.ndarray, fill: float | None) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    if fill is not None:
        out[np.isclose(out, fill) | (np.abs(out) > FILL_THRESHOLD)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())
