"""Minimal title cards for composite movie section breaks."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lib.video_encode import encode_mp4

BG = (13, 17, 23)
TITLE_COLOR = (240, 246, 252)
SUBTITLE_COLOR = (201, 209, 217)

_TITLE_SIZE = 44
_SUBTITLE_SIZE = 28
_LINE_GAP = 28

_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ("/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial.ttf"),
    ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = 0 if bold else 1
    for bold_path, regular_path in _FONT_CANDIDATES:
        path = bold_path if bold else regular_path
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_title_card_png(
    path: Path,
    *,
    title: str,
    subtitle: str,
    width: int = 1280,
    height: int = 720,
) -> None:
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title_font = _load_font(_TITLE_SIZE, bold=True)
    subtitle_font = _load_font(_SUBTITLE_SIZE)

    title_w, title_h = _text_size(draw, title, title_font)
    sub_w, sub_h = _text_size(draw, subtitle, subtitle_font)
    block_h = title_h + _LINE_GAP + sub_h
    y0 = (height - block_h) // 2

    draw.text(
        ((width - title_w) // 2, y0),
        title,
        fill=TITLE_COLOR,
        font=title_font,
    )
    draw.text(
        ((width - sub_w) // 2, y0 + title_h + _LINE_GAP),
        subtitle,
        fill=SUBTITLE_COLOR,
        font=subtitle_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def render_title_card_mp4(
    title: str,
    subtitle: str,
    output: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 10,
    duration_sec: float = 3.0,
) -> Path:
    n_frames = max(1, int(round(duration_sec * fps)))
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="title_card_") as tmp:
        frame_dir = Path(tmp)
        card_png = frame_dir / "card.png"
        render_title_card_png(
            card_png,
            title=title,
            subtitle=subtitle,
            width=width,
            height=height,
        )
        for i in range(n_frames):
            dest = frame_dir / f"card_{i:04d}.png"
            dest.write_bytes(card_png.read_bytes())

        if not encode_mp4(frame_dir, "card_%04d.png", output, fps):
            raise RuntimeError(
                "Failed to encode title card MP4 (ffmpeg required). "
                "Install with: brew install ffmpeg"
            )

    return output
