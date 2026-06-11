"""PIL annotations for rendered movie frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class FrameOverlay:
    title: str
    subtitle: str
    timestamp: str
    frame_label: str
    region_label: str
    notes: str
    elev_vmin: float
    elev_vmax: float
    slab_label: str
    slab_vmin: float
    slab_vmax: float
    slab_color_mode: str = "cloud"
    legend_mode: str = "default"

    @property
    def clt_vmin(self) -> float:
        return self.slab_vmin

    @property
    def clt_vmax(self) -> float:
        return self.slab_vmax


COMPARE_LEGEND_ITEMS = (
    ("Actual deck", (46, 199, 242)),
    ("Predicted deck", (250, 148, 30)),
)

FEATURE_OVERLAY_LEGEND_ITEMS = (
    ("Predicted deck", (250, 148, 30)),
    ("Low cloud (LCC)", (210, 228, 250)),
    ("2 m temp", (240, 120, 80)),
    ("10 m wind", (230, 235, 245)),
)

PREDICTED_DECK_SWATCH = ("Predicted deck", (250, 148, 30))


def _terrain_color(t: float) -> tuple[int, int, int]:
    if t < 0.35:
        r, g, b = 0.25 + 0.35 * t, 0.45 + 0.25 * t, 0.18 + 0.1 * t
    elif t < 0.7:
        u = (t - 0.35) / 0.35
        r, g, b = 0.55 + 0.15 * u, 0.52 + 0.08 * u, 0.32 + 0.12 * u
    else:
        u = (t - 0.7) / 0.3
        r, g, b = 0.7 + 0.25 * u, 0.6 + 0.3 * u, 0.44 + 0.5 * u
    return int(r * 255), int(g * 255), int(b * 255)


def _cloud_color(t: float) -> tuple[int, int, int, int]:
    alpha = max(0.0, min(1.0, (t - 0.15) / 0.85))
    r = 0.75 + 0.25 * t
    g = 0.82 + 0.15 * t
    b = 0.95
    return int(r * 255), int(g * 255), int(b * 255), int(alpha * 220)


def _pressure_color(t: float) -> tuple[int, int, int]:
    # low height = cool blue, high = warm white
    r = 0.35 + 0.65 * t
    g = 0.55 + 0.4 * t
    b = 0.95 - 0.35 * t
    return int(r * 255), int(g * 255), int(b * 255)


def _wind_color(t: float) -> tuple[int, int, int]:
    # calm green -> moderate yellow -> strong red
    if t < 0.5:
        u = t / 0.5
        r, g, b = 0.2 + 0.6 * u, 0.65 + 0.25 * u, 0.25
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 0.8 + 0.2 * u, 0.9 - 0.5 * u, 0.25 - 0.15 * u
    return int(r * 255), int(g * 255), int(b * 255)


def _t2m_color(t: float) -> tuple[int, int, int]:
    r = 0.12 + 0.88 * t
    g = 0.35 + 0.45 * (1.0 - abs(t - 0.5) * 2.0)
    b = 0.95 - 0.8 * t
    return int(r * 255), int(g * 255), int(b * 255)


def _slab_color(t: float, mode: str) -> tuple[int, int, int]:
    if mode == "pressure":
        return _pressure_color(t)
    if mode == "wind":
        return _wind_color(t)
    if mode == "t2m":
        return _t2m_color(t)
    return _cloud_color(t)[:3]


def _draw_colorbar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    vmin: float,
    vmax: float,
    label: str,
    *,
    mode: str,
) -> None:
    for i in range(width):
        t = i / max(width - 1, 1)
        if mode == "terrain":
            color = _terrain_color(t)
        else:
            color = _slab_color(t, mode)
        draw.line([(x + i, y), (x + i, y + height)], fill=color)
    draw.rectangle([x, y, x + width, y + height], outline=(200, 210, 220))
    draw.text((x, y - 16), label, fill=(230, 235, 245))
    if mode == "cloud":
        draw.text((x, y + height + 2), f"{vmin:.0f}", fill=(180, 190, 200))
        draw.text((x + width - 28, y + height + 2), f"{vmax:.0f}", fill=(180, 190, 200))
    elif mode == "t2m":
        draw.text((x, y + height + 2), f"{vmin:.1f}", fill=(180, 190, 200))
        draw.text((x + width - 36, y + height + 2), f"{vmax:.1f}", fill=(180, 190, 200))
    else:
        draw.text((x, y + height + 2), f"{vmin:.0f}", fill=(180, 190, 200))
        draw.text((x + width - 36, y + height + 2), f"{vmax:.0f}", fill=(180, 190, 200))


def _draw_compare_legend(draw: ImageDraw.ImageDraw, img_width: int, img_height: int) -> None:
    """Swatch legend for inversion compare movies (two decks only)."""
    _draw_swatch_legend(draw, img_width, img_height, COMPARE_LEGEND_ITEMS)


def _draw_swatch_legend(
    draw: ImageDraw.ImageDraw,
    img_width: int,
    img_height: int,
    items: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    swatch_w, swatch_h = 36, 18
    gap_x = 24
    row = items
    total_w = len(row) * swatch_w + (len(row) - 1) * gap_x
    x0 = img_width - total_w - 28
    y0 = img_height - 88

    draw.text((x0, y0 - 18), "Legend", fill=(240, 246, 252))

    x = x0
    for label, rgb in row:
        draw.rectangle(
            [x, y0, x + swatch_w, y0 + swatch_h],
            fill=rgb,
            outline=(230, 235, 245),
            width=1,
        )
        draw.text((x, y0 + swatch_h + 4), label, fill=(220, 228, 238))
        x += swatch_w + gap_x


def _draw_feature_overlay_legend(draw: ImageDraw.ImageDraw, img_width: int, img_height: int) -> None:
    _draw_swatch_legend(draw, img_width, img_height, FEATURE_OVERLAY_LEGEND_ITEMS)


def _draw_feature_single_legend(
    draw: ImageDraw.ImageDraw,
    img_width: int,
    img_height: int,
    *,
    feature_label: str,
    feature_vmin: float,
    feature_vmax: float,
    feature_color_mode: str,
) -> None:
    """Predicted-deck swatch plus fixed-scale feature colorbar."""
    swatch_w, swatch_h = 36, 18
    bar_w, bar_h = 180, 12
    y0 = img_height - 88
    x_deck = img_width - swatch_w - bar_w - 48
    x_bar = img_width - bar_w - 28

    draw.text((x_deck, y0 - 18), "Legend", fill=(240, 246, 252))
    label, rgb = PREDICTED_DECK_SWATCH
    draw.rectangle(
        [x_deck, y0, x_deck + swatch_w, y0 + swatch_h],
        fill=rgb,
        outline=(230, 235, 245),
        width=1,
    )
    draw.text((x_deck, y0 + swatch_h + 4), label, fill=(220, 228, 238))

    _draw_colorbar(
        draw, x_bar, y0 + 2, bar_w, bar_h,
        feature_vmin, feature_vmax, feature_label,
        mode=feature_color_mode,
    )


def annotate_frame_png(path: Path, overlay: FrameOverlay) -> None:
    img = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    panel_h = 118
    panel = Image.new("RGBA", (img.width, panel_h), (7, 11, 20, 210))
    img.paste(panel, (0, 0), panel)
    draw = ImageDraw.Draw(img)

    draw.text((18, 12), overlay.title, fill=(240, 246, 252))
    draw.text((18, 36), overlay.subtitle, fill=(201, 209, 217))
    draw.text((18, 58), overlay.timestamp, fill=(201, 209, 217))
    draw.text((18, 80), overlay.frame_label, fill=(139, 148, 158))

    draw.text((18, img.height - 52), overlay.region_label, fill=(201, 209, 217))
    draw.text((18, img.height - 32), overlay.notes, fill=(139, 148, 158))

    bar_y = img.height - 78
    if overlay.legend_mode == "compare":
        _draw_compare_legend(draw, img.width, img.height)
    elif overlay.legend_mode == "feature_overlay":
        _draw_feature_overlay_legend(draw, img.width, img.height)
    elif overlay.legend_mode == "feature_single":
        _draw_feature_single_legend(
            draw,
            img.width,
            img.height,
            feature_label=overlay.slab_label,
            feature_vmin=overlay.slab_vmin,
            feature_vmax=overlay.slab_vmax,
            feature_color_mode=overlay.slab_color_mode,
        )
    else:
        _draw_colorbar(
            draw, img.width - 360, bar_y, 150, 12,
            overlay.elev_vmin, overlay.elev_vmax, "Elevation (m)", mode="terrain",
        )
        _draw_colorbar(
            draw, img.width - 190, bar_y, 150, 12,
            overlay.slab_vmin, overlay.slab_vmax, overlay.slab_label,
            mode=overlay.slab_color_mode,
        )

    img.convert("RGB").save(path)
