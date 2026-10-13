"""Where everything lives on disk. One function per kind of file, so paths never drift."""
from __future__ import annotations

import datetime as dt
from pathlib import Path


def stamp(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_stamp(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def raw_path(root: Path, camera: str, ts: dt.datetime) -> Path:
    u = ts.astimezone(dt.timezone.utc)
    return root / "raw" / camera / u.strftime("%Y/%m/%d") / f"{stamp(u)}.jpg"


def crop_path(root: Path, plant_id: str, ts: dt.datetime) -> Path:
    u = ts.astimezone(dt.timezone.utc)
    return root / "plants" / plant_id / u.strftime("%Y/%m/%d") / f"{stamp(u)}.jpg"
