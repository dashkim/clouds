"""Phenomenological cloud-inversion detection on 2D lon/lat grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from lib.meteo_utils import pressure_to_height_msl

VALLEY_Z_OFFSET_M = 200.0

InversionMode = Literal["deck_only", "phenomenological"]


@dataclass
class InversionParams:
    """Tunable inversion detection settings."""

    mode: InversionMode = "phenomenological"
    z_min_m: float = 800.0
    cbh_margin_m: float = 100.0
    deck_cover_min: float = 20.0
    clear_max: float = 25.0
    valley_cloud_min: float = 50.0
    valley_z_offset_m: float = VALLEY_Z_OFFSET_M
    use_neighborhood: bool = True

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "z_min_m": self.z_min_m,
            "cbh_margin_m": self.cbh_margin_m,
            "deck_cover_min": self.deck_cover_min,
            "clear_max": self.clear_max,
            "valley_cloud_min": self.valley_cloud_min,
            "valley_z_offset_m": self.valley_z_offset_m,
            "use_neighborhood": self.use_neighborhood,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InversionParams:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def cover_to_percent(values: np.ndarray) -> np.ndarray:
    """Normalize cloud cover to 0–100 % (ERA5 uses 0–1, ICON uses 0–100)."""
    arr = np.asarray(values, dtype=np.float64)
    out = arr.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out.astype(np.float32)
    if float(np.nanmax(out[finite])) <= 1.0 + 1e-6:
        out[finite] *= 100.0
    return out.astype(np.float32)


def cloud_base_from_ccb(ccb_pa: np.ndarray) -> np.ndarray:
    """ICON cloud-base pressure (Pa) → height MSL (m)."""
    return pressure_to_height_msl(ccb_pa)


def cbh_agl_to_msl(cbh_agl: np.ndarray, terrain_z: np.ndarray) -> np.ndarray:
    """ERA5 cloud base height is above ground level → convert to MSL (m)."""
    cbh = np.asarray(cbh_agl, dtype=np.float64)
    z = np.asarray(terrain_z, dtype=np.float64)
    out = z + cbh
    bad = ~np.isfinite(z) | ~np.isfinite(cbh) | (cbh <= 0.0)
    out[bad] = np.nan
    return out.astype(np.float32)


def inversion_fog_deck_msl(
    terrain_z: np.ndarray,
    cbh_agl: np.ndarray,
    cover: np.ndarray,
    *,
    cover_min: float = 20.0,
) -> float:
    """
    Single MSL height for the horizontal inversion fog deck.

    Uses cloud-base MSL in lowland cloudy cells so the layer cuts through
    mid-elevations and peaks rise above it.
    """
    z = np.asarray(terrain_z, dtype=np.float64)
    msl = cbh_agl_to_msl(cbh_agl, z)
    cover_pct = cover_to_percent(cover)
    lowland = z <= np.nanpercentile(z[np.isfinite(z)], 55)
    cloudy = np.isfinite(msl) & (cover_pct >= cover_min)
    ref = msl[lowland & cloudy]
    if np.any(ref):
        return float(np.nanpercentile(ref, 42))
    valid = msl[np.isfinite(msl)]
    return float(np.nanpercentile(valid, 50)) if valid.size else 0.0


def predicted_fog_deck_msl(
    terrain_z: np.ndarray,
    cbh_agl: np.ndarray,
    cover: np.ndarray,
    predicted_mask: np.ndarray,
    actual_mask: np.ndarray | None = None,
    *,
    z_min_m: float = 800.0,
    cover_min: float = 20.0,
) -> float:
    """
    Estimated fog-deck MSL from ML-predicted inversion vs actual.

    When the model over-predicts ridge inversion, the predicted deck is drawn
    lower; when it under-predicts, the deck is drawn higher.
    """
    z = np.asarray(terrain_z, dtype=np.float64)
    actual_deck = inversion_fog_deck_msl(z, cbh_agl, cover, cover_min=cover_min)
    ridge = np.isfinite(z) & (z >= z_min_m)
    if not np.any(ridge):
        return actual_deck

    pred = np.asarray(predicted_mask, dtype=np.float64)
    pred_frac = float(np.mean(pred[ridge] >= 0.5))
    if actual_mask is not None:
        act = np.asarray(actual_mask, dtype=np.float64)
        act_frac = float(np.mean(act[ridge] >= 0.5))
    else:
        act_frac = 0.0

    z_span = float(np.nanmax(z[ridge]) - np.nanmin(z[np.isfinite(z)]))
    z_span = max(z_span, 200.0)
    offset = (pred_frac - act_frac) * z_span * 0.18
    return float(actual_deck - offset)


def _valley_cloud_below(
    terrain_z: np.ndarray,
    low_cover: np.ndarray,
    medium_cover: np.ndarray,
    *,
    valley_z_offset_m: float,
    valley_cloud_min: float,
    use_neighborhood: bool,
) -> np.ndarray:
    """True where lower terrain in the patch is cloudy enough."""
    low = cover_to_percent(low_cover)
    med = cover_to_percent(medium_cover)
    layer_cover = np.maximum(low, med)

    if use_neighborhood:
        footprint = np.ones((5, 5), dtype=bool)
        layer_cover = ndimage.maximum_filter(layer_cover, footprint=footprint, mode="nearest")

    z = np.asarray(terrain_z, dtype=np.float64)
    valid_z = np.isfinite(z)
    result = np.zeros(z.shape, dtype=bool)
    if not np.any(valid_z):
        return result

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    if z_max - z_min < valley_z_offset_m:
        below = valid_z & (z <= np.nanpercentile(z[valid_z], 40))
        if np.any(below):
            mean_below = float(np.nanmean(layer_cover[below]))
            if mean_below >= valley_cloud_min:
                result[valid_z] = True
        return result

    thresholds = np.linspace(z_min + valley_z_offset_m, z_max, 16)
    for thresh in thresholds:
        below = valid_z & (z < thresh - valley_z_offset_m)
        if np.count_nonzero(below) < 3:
            continue
        mean_below = float(np.nanmean(layer_cover[below]))
        if mean_below >= valley_cloud_min:
            result[valid_z & (z >= thresh)] = True
    return result


def _peak_above_cloud_base(
    terrain_z: np.ndarray,
    cloud_base_m: np.ndarray,
    *,
    cbh_margin_m: float,
    cloud_cover: np.ndarray | None = None,
) -> np.ndarray:
    """Peak terrain above the regional fog deck (ERA5 cbh is AGL)."""
    z = np.asarray(terrain_z, dtype=np.float64)
    cbh = np.asarray(cloud_base_m, dtype=np.float64)
    valid = np.isfinite(z) & np.isfinite(cbh) & (cbh > 0.0)
    if not np.any(valid):
        return np.zeros(z.shape, dtype=bool)

    if cloud_cover is not None:
        deck = inversion_fog_deck_msl(z, cbh, cloud_cover)
    else:
        msl = cbh_agl_to_msl(cbh, z)
        deck = float(np.nanpercentile(msl[valid], 55))

    return valid & (z > deck + cbh_margin_m)


def detect_inversion_mask(
    terrain_z: np.ndarray,
    *,
    cloud_cover: np.ndarray,
    low_cover: np.ndarray | None = None,
    medium_cover: np.ndarray | None = None,
    cloud_base_m: np.ndarray | None = None,
    z_min_m: float = 1000.0,
    clear_max: float = 25.0,
    valley_cloud_min: float = 50.0,
    cbh_margin_m: float = 100.0,
    valley_z_offset_m: float = VALLEY_Z_OFFSET_M,
    use_neighborhood: bool = True,
    params: InversionParams | None = None,
) -> np.ndarray:
    """
    Detect inversion on ridge cells.

    ``deck_only``: ridge terrain above regional fog deck (peaks above cloud layer).
    ``phenomenological``: ridge above deck, locally clear, and valleys cloudy below.

    Returns float32 array: 1.0 = inversion, 0.0 = not, nan = invalid terrain.
    """
    if params is None:
        params = InversionParams(
            mode="phenomenological",
            z_min_m=z_min_m,
            cbh_margin_m=cbh_margin_m,
            clear_max=clear_max,
            valley_cloud_min=valley_cloud_min,
            valley_z_offset_m=valley_z_offset_m,
            use_neighborhood=use_neighborhood,
        )

    z = np.asarray(terrain_z, dtype=np.float64)
    cover = cover_to_percent(cloud_cover)

    out = np.full(z.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(z)
    out[~valid] = np.nan

    ridge = valid & (z >= params.z_min_m)

    has_cbh = cloud_base_m is not None and np.any(np.isfinite(cloud_base_m))
    if has_cbh:
        above_deck = _peak_above_cloud_base(
            z,
            cloud_base_m,
            cbh_margin_m=params.cbh_margin_m,
            cloud_cover=cloud_cover,
        )
    else:
        above_deck = cover < params.clear_max

    if params.mode == "deck_only":
        inverted = ridge & above_deck
    else:
        if low_cover is None:
            low_cover = cover
        if medium_cover is None:
            medium_cover = np.zeros_like(cover)

        valley_cloudy = _valley_cloud_below(
            z,
            low_cover,
            medium_cover,
            valley_z_offset_m=params.valley_z_offset_m,
            valley_cloud_min=params.valley_cloud_min,
            use_neighborhood=params.use_neighborhood,
        )
        inverted = ridge & above_deck & valley_cloudy

    out[valid] = 0.0
    out[inverted] = 1.0
    return out


def inversion_fraction(mask: np.ndarray) -> float:
    """Fraction of valid grid cells marked inverted."""
    valid = np.isfinite(mask)
    if not np.any(valid):
        return 0.0
    return float(np.sum(mask[valid] >= 0.5) / np.sum(valid))


def ridge_inversion_fraction(
    mask: np.ndarray,
    terrain_z: np.ndarray,
    *,
    z_min_m: float = 800.0,
) -> float:
    """Fraction of ridge cells (z >= z_min_m) marked inverted."""
    z = np.asarray(terrain_z, dtype=np.float64)
    ridge = np.isfinite(z) & (z >= z_min_m)
    if not np.any(ridge):
        return 0.0
    slab = np.asarray(mask, dtype=np.float32)
    valid = ridge & np.isfinite(slab)
    if not np.any(valid):
        return 0.0
    return float(np.sum(slab[valid] >= 0.5) / np.sum(valid))


def ridge_above_deck_fraction(
    terrain_z: np.ndarray,
    cloud_base_m: np.ndarray,
    *,
    z_min_m: float = 1000.0,
    margin_m: float = 100.0,
    cloud_cover: np.ndarray | None = None,
) -> float:
    """Fraction of ridge cells whose terrain exceeds the fog deck + margin."""
    z = np.asarray(terrain_z, dtype=np.float64)
    cbh = np.asarray(cloud_base_m, dtype=np.float64)
    ridge = np.isfinite(z) & (z >= z_min_m)
    if not np.any(ridge):
        return 0.0
    above = ridge & _peak_above_cloud_base(
        z, cbh, cbh_margin_m=margin_m, cloud_cover=cloud_cover,
    )
    return float(np.sum(above) / np.sum(ridge))
