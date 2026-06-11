#!/usr/bin/env python3
"""Render and stitch the composite cloud-inversion movie."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import CLOUDS_ROOT, MODELS_ARTIFACTS_DIR
from lib.title_card import render_title_card_mp4
from lib.video_encode import concat_mp4, find_ffmpeg, require_ffmpeg

COMPOSITE_WIDTH = 1280
COMPOSITE_HEIGHT = 720
COMPOSITE_FPS = 10
COMPOSITE_FPS_2D = 20
ERA5_START = "2013-01-01T18:00"
ERA5_FRAMES = 12
INTERP_STEPS = 4
RENDER_MULTIPLIER = 4
CLOUD_TIME_SCALE = 0.35

SEGMENTS_DIR = CLOUDS_ROOT / "output" / "composite" / "segments"
DEFAULT_OUTPUT = CLOUDS_ROOT / "output" / "composite" / "cloud_inversion_composite.mp4"

CAMERA_3D = [
    "--region", "east_core",
    "--width", str(COMPOSITE_WIDTH),
    "--height", str(COMPOSITE_HEIGHT),
    "--fps", str(COMPOSITE_FPS),
    "--elevation-deg", "24",
    "--orbit-deg", "40",
]

ERA5_ANIM = [
    "--era5-start", ERA5_START,
    "--era5-frames", str(ERA5_FRAMES),
    "--interp-steps", str(INTERP_STEPS),
    "--render-multiplier", str(RENDER_MULTIPLIER),
    "--cloud-time-scale", str(CLOUD_TIME_SCALE),
    "--fps", str(COMPOSITE_FPS),
]

COMPOSITE_SECTIONS: list[dict[str, Any]] = [
    {
        "type": "card",
        "id": "card_2d_data",
        "title": "ICON HD(CP)² — 2D lon/lat grid",
        "subtitle": "Total cloud cover over terrain · 121 frames · 26 Apr 2013",
    },
    {"type": "clip", "id": "germany_clouds", "render": "country"},
    {"type": "clip", "id": "alps_patch_clouds", "render": "alps_patch"},
    {
        "type": "card",
        "id": "card_3d_mesh",
        "title": "Pseudo-3D terrain mesh",
        "subtitle": (
            "ICON elevation in local meters · 2D fields on a semi-transparent slab · "
            "10× vertical exaggeration"
        ),
    },
    {"type": "clip", "id": "east_patch_3d_clouds", "render": "patch_3d_clt"},
    {"type": "clip", "id": "east_patch_3d_wind", "render": "patch_3d_wind"},
    {
        "type": "card",
        "id": "card_era5",
        "title": "ERA5 reanalysis labels",
        "subtitle": "Hourly cloud-base height and low cloud cover for inversion detection",
    },
    {"type": "clip", "id": "east_core_patch_3d_inversion", "render": "patch_3d_inversion"},
    {"type": "clip", "id": "east_core_inversion_compare", "render": "compare"},
    {
        "type": "card",
        "id": "card_features",
        "title": "Ridge regression inputs",
        "subtitle": "ML predicted fog deck + LCC, 2 m temperature, and 10 m wind",
    },
    {"type": "clip", "id": "feature_overlay_lcc", "render": "feature_lcc"},
    {"type": "clip", "id": "feature_overlay_t2m", "render": "feature_t2m"},
    {"type": "clip", "id": "feature_overlay_wind", "render": "feature_wind"},
]


def segment_path(section_id: str) -> Path:
    return SEGMENTS_DIR / f"{section_id}.mp4"


def check_prerequisites(*, need_ml: bool = True) -> None:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Activate the project venv and run:\n"
            "  pip install -r requirements.txt"
        ) from exc

    if need_ml:
        try:
            import joblib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "joblib is required for compare/feature segments. Run:\n"
                "  pip install -r requirements.txt"
            ) from exc

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        )
    print(f"Using ffmpeg: {ffmpeg}")

    if need_ml and not MODELS_ARTIFACTS_DIR.joinpath("east_core_predicted_masks.npz").is_file():
        raise FileNotFoundError(
            "Missing models/artifacts/east_core_predicted_masks.npz — "
            "required for compare and feature-overlay segments."
        )


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or CLOUDS_ROOT)


def render_card(section: dict[str, Any], *, duration_sec: float) -> Path:
    output = segment_path(section["id"])
    render_title_card_mp4(
        section["title"],
        section["subtitle"],
        output,
        width=COMPOSITE_WIDTH,
        height=COMPOSITE_HEIGHT,
        fps=COMPOSITE_FPS,
        duration_sec=duration_sec,
    )
    print(f"Wrote {output}")
    return output


def render_clip(section: dict[str, Any]) -> Path:
    output = segment_path(section["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    render_key = section["render"]
    py = sys.executable

    if render_key == "country":
        _run([
            py, "scripts/render_country_cloud_movie.py",
            "--output", str(output),
            "--fps", str(COMPOSITE_FPS_2D),
            "--width", str(COMPOSITE_WIDTH),
            "--height", str(COMPOSITE_HEIGHT),
        ])
    elif render_key == "alps_patch":
        _run([
            py, "scripts/render_alps_patch_movie.py",
            "--output", str(output),
            "--fps", str(COMPOSITE_FPS_2D),
            "--width", str(COMPOSITE_WIDTH),
            "--height", str(COMPOSITE_HEIGHT),
        ])
    elif render_key == "patch_3d_clt":
        _run([
            py, "scripts/render_patch_3d_cloud_movie.py",
            *CAMERA_3D,
            "--slab-mode", "clt",
            "--orbit-start-deg", "-50",
            "--output", str(output),
        ])
    elif render_key == "patch_3d_wind":
        _run([
            py, "scripts/render_patch_3d_cloud_movie.py",
            *CAMERA_3D,
            "--slab-mode", "wind",
            "--orbit-start-deg", "115",
            "--output", str(output),
        ])
    elif render_key == "patch_3d_inversion":
        _run([
            py, "scripts/render_patch_3d_cloud_movie.py",
            *CAMERA_3D,
            "--slab-mode", "inversion",
            "--label-source", "era5",
            "--orbit-start-deg", "-50",
            *ERA5_ANIM,
            "--output", str(output),
        ])
    elif render_key == "compare":
        _run([
            py, "models/render/compare.py",
            *CAMERA_3D,
            *ERA5_ANIM,
            "--output", str(output),
        ])
    elif render_key in ("feature_lcc", "feature_t2m", "feature_wind"):
        feature = render_key.removeprefix("feature_")
        _run([
            py, "models/render/feature_overlay.py",
            *CAMERA_3D,
            *ERA5_ANIM,
            "--feature", feature,
            "--output", str(output),
        ])
    else:
        raise ValueError(f"Unknown render key: {render_key}")

    if not output.is_file():
        raise FileNotFoundError(f"Expected render output missing: {output}")
    print(f"Wrote {output}")
    return output


def render_sections(
    sections: list[dict[str, Any]],
    *,
    card_duration: float,
    skip_existing: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for section in sections:
        sid = section["id"]
        output = segment_path(sid)
        if skip_existing and output.is_file():
            print(f"\n=== {sid} (skip — exists) ===")
            paths.append(output)
            continue
        print(f"\n=== {sid} ===")
        if section["type"] == "card":
            paths.append(render_card(section, duration_sec=card_duration))
        else:
            paths.append(render_clip(section))
    return paths


def stitch_segments(segments: list[Path], output: Path) -> Path:
    require_ffmpeg()
    concat_mp4(segments, output)
    print(f"Wrote {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final composite MP4 path",
    )
    parser.add_argument(
        "--card-duration",
        type=float,
        default=3.0,
        help="Seconds per title card (default: 3)",
    )
    parser.add_argument(
        "--render-segments",
        action="store_true",
        help="Re-render all segments (title cards + clips)",
    )
    parser.add_argument(
        "--stitch-only",
        action="store_true",
        help="Concatenate existing segments only (skip render)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip segment renders when the MP4 already exists (resume after failure)",
    )
    args = parser.parse_args()

    do_render = args.render_segments or not args.stitch_only
    do_stitch = not args.render_segments or not args.stitch_only
    if args.render_segments and args.stitch_only:
        do_render = True
        do_stitch = True

    if do_render:
        print("Rendering composite segments…")
        check_prerequisites(need_ml=True)
        render_sections(
            COMPOSITE_SECTIONS,
            card_duration=args.card_duration,
            skip_existing=args.skip_existing,
        )

    segment_paths = [segment_path(s["id"]) for s in COMPOSITE_SECTIONS]
    missing = [p for p in segment_paths if not p.is_file()]
    if missing and do_stitch:
        names = ", ".join(p.name for p in missing)
        print(f"Missing segments: {names}", file=sys.stderr)
        return 1

    if do_stitch:
        print("\nStitching composite movie…")
        require_ffmpeg()
        stitch_segments(segment_paths, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
