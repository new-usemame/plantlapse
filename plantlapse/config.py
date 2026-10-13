"""Every setting comes from the environment. Nothing about any particular home lives in code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Config:
    # The stream to sample. Any URL ffmpeg can open (RTSP restream, camera RTSP, HTTP MJPEG...).
    stream_url: str
    camera: str
    # Optional still-image URL (e.g. a camera's own JPEG snapshot). Tried before the stream: a camera's
    # JPEG is encoded from the sensor frame, while a stream keyframe carries the video's compression.
    snapshot_url: str | None
    # Frames (raw + per-plant crops + renders) live here. Can be a network share.
    data_dir: Path
    # The tracker database lives here. Keep it on local disk: SQLite on NFS is a bad idea.
    state_dir: Path
    interval_min: int
    api_token: str | None
    burst: int
    crop_px: int
    torch_threads: int
    latlon: tuple[float, float] | None = None  # decides day/night by the sun; deploy env only

    @property
    def db_path(self) -> Path:
        return self.state_dir / "plantlapse.db"


def load() -> Config:
    return Config(
        stream_url=_env("PLANTLAPSE_STREAM_URL", "") or "",
        camera=_env("PLANTLAPSE_CAMERA", "cam") or "cam",
        snapshot_url=_env("PLANTLAPSE_SNAPSHOT_URL"),
        data_dir=Path(_env("PLANTLAPSE_DATA_DIR", "/data") or "/data"),
        state_dir=Path(_env("PLANTLAPSE_STATE_DIR", "/state") or "/state"),
        interval_min=int(_env("PLANTLAPSE_INTERVAL_MIN", "15") or 15),
        api_token=_env("PLANTLAPSE_TOKEN"),
        burst=int(_env("PLANTLAPSE_BURST", "3") or 3),
        crop_px=int(_env("PLANTLAPSE_CROP_PX", "768") or 768),
        torch_threads=int(_env("PLANTLAPSE_TORCH_THREADS", "4") or 4),
        latlon=_latlon(_env("PLANTLAPSE_LATLON")),
    )


def _latlon(v: str | None) -> tuple[float, float] | None:
    if not v:
        return None
    lat, lon = (float(x) for x in v.split(","))
    return (lat, lon)
