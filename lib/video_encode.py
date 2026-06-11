"""Encode PNG frame sequences and GIFs for movie output."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

_FFMPEG_CANDIDATES = (
    "ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
)


def find_ffmpeg() -> str | None:
    """Return ffmpeg executable path, checking common install locations."""
    for candidate in _FFMPEG_CANDIDATES:
        if candidate == "ffmpeg":
            found = shutil.which("ffmpeg")
            if found:
                return found
        elif Path(candidate).is_file():
            return candidate
    return None


def require_ffmpeg() -> str:
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg\n"
            "Then ensure /opt/homebrew/bin is on your PATH."
        )
    return exe


def encode_mp4(frame_dir: Path, pattern: str, output: Path, fps: int) -> bool:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / pattern),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output),
    ]
    env = os.environ.copy()
    ffmpeg_dir = str(Path(ffmpeg).parent)
    if ffmpeg_dir not in env.get("PATH", "").split(":"):
        env["PATH"] = f"{ffmpeg_dir}:{env.get('PATH', '')}"
    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode(errors="replace"), file=sys.stderr)
        return False


def encode_gif(frames: list[Path], output: Path, fps: int) -> None:
    images = [Image.open(p).convert("RGB") for p in frames]
    duration_ms = max(1, int(1000 / fps))
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for img in images:
        img.close()


def concat_mp4(segments: list[Path], output: Path) -> None:
    """Concatenate MP4 segments with identical codec settings (stream copy)."""
    ffmpeg = require_ffmpeg()
    if not segments:
        raise ValueError("No segments to concatenate")
    for path in segments:
        if not path.is_file():
            raise FileNotFoundError(f"Missing segment: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.with_suffix(".concat.txt")
    try:
        lines = [f"file '{p.resolve()}'" for p in segments]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode(errors="replace"), file=sys.stderr)
        raise
    finally:
        list_path.unlink(missing_ok=True)
