"""Resolve paths to local NetCDF files under data/."""
from __future__ import annotations

import argparse
from pathlib import Path

CLOUDS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CLOUDS_ROOT / "data"
MODELS_DIR = CLOUDS_ROOT / "models"
MODELS_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

_SEARCH_PATTERNS = {
    "icon_dom": ("3d_icon_dom*.nc",),
    "lonlat": ("2d_lonlat*.nc",),
}

_ERA5_PREFERENCE = (
    "2013_germany.grib",
    "*_germany*.grib",
)


def _resolve_explicit(explicit: str) -> Path:
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NetCDF not found: {path}")
    return path


def _find_by_patterns(kind: str) -> Path:
    candidates: list[Path] = []
    for pattern in _SEARCH_PATTERNS[kind]:
        candidates.extend(sorted(DATA_DIR.glob(pattern)))

    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(
            f"Multiple {kind} files found ({names}). Pass --input explicitly."
        )
    patterns = ", ".join(_SEARCH_PATTERNS[kind])
    raise FileNotFoundError(
        f"No {kind} NetCDF ({patterns}) under {DATA_DIR}/. "
        "Place the file there or pass --input."
    )


def find_icon_dom_nc(explicit: str | None = None) -> Path:
    if explicit:
        return _resolve_explicit(explicit)
    return _find_by_patterns("icon_dom")


def find_lonlat_nc(explicit: str | None = None) -> Path:
    if explicit:
        return _resolve_explicit(explicit)
    return _find_by_patterns("lonlat")


def _resolve_grib_explicit(explicit: str) -> Path:
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GRIB not found: {path}")
    return path


def find_era5_grib(explicit: str | None = None) -> Path:
    """Prefer full-year Germany ERA5, then April sample."""
    if explicit:
        return _resolve_grib_explicit(explicit)
    for pattern in _ERA5_PREFERENCE:
        if "*" in pattern:
            candidates = sorted(DATA_DIR.glob(pattern))
            if candidates:
                return candidates[0].resolve()
        else:
            path = DATA_DIR / pattern
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(
        f"No ERA5 GRIB under {DATA_DIR}/. "
        "Place 2013_germany.grib there or pass --era5-grib."
    )


def add_era5_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--era5-grib",
        help="Path to ERA5 GRIB (default: auto-detect under data/)",
    )


def add_input_arg(parser: argparse.ArgumentParser, *, kind: str = "icon_dom") -> None:
    label = "3d_icon_dom*.nc" if kind == "icon_dom" else "2d_lonlat*.nc"
    parser.add_argument(
        "--input",
        "-i",
        help=f"Path to {label} (default: auto-detect under data/)",
    )
