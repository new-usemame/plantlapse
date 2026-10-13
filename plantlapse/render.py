"""Turn a plant's frames into an MP4. Cached by the exact frame list and options."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def render(frames: list[Path], out_dir: Path, fps: int = 24, size: int = 1080,
           deflicker: bool = True, label: str = "") -> Path:
    if not frames:
        raise ValueError("no frames")
    key = hashlib.sha256(
        # mtime too: a reframe rewrites crops in place, and the old clip must not be served for them.
        ("|".join(f"{f}:{f.stat().st_mtime_ns}" for f in frames) + f"|{fps}|{size}|{deflicker}|{label}")
        .encode()).hexdigest()[:20]
    out = out_dir / f"{key}.mp4"
    if out.exists():
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for f in frames:
            lst.write(f"file '{f}'\nduration {1 / fps:.6f}\n")
        lst.write(f"file '{frames[-1]}'\n")
        list_path = lst.name
    vf = [f"scale={size}:{size}:flags=lanczos"]
    if deflicker and len(frames) >= 5:
        vf.append("deflicker=mode=pm:size=5")
    vf.append("format=yuv420p")
    tmp = out.with_suffix(".part.mp4")
    try:
        subprocess.run(
            ["nice", "-n", "10", "ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-vf", ",".join(vf), "-r", str(max(fps, 24)), "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-threads", "2", "-movflags", "+faststart", str(tmp)],
            check=True, capture_output=True, timeout=900)
        os.replace(tmp, out)
    finally:
        os.unlink(list_path)
        tmp.unlink(missing_ok=True)
    return out
