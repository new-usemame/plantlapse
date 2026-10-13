"""Backfill raw frames from Frigate's recordings before they expire.

Frigate keeps continuous recordings for a limited window. For every capture slot
(default every 15 minutes) inside that window, pull one frame out of the segment
that covers it and store it exactly where the live capture would have.

    python -m plantlapse.backfill --recordings /recordings --camera porch --out /data
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path

from .storage import raw_path


def segments_for_hour(hour_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in hour_dir.glob("*.mp4"):
        try:
            mm, ss = p.stem.split(".")
            out.append((int(mm) * 60 + int(ss), p))
        except ValueError:
            continue
    return sorted(out)


def extract(segment: Path, offset_s: float, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part.jpg")
    r = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{offset_s:.2f}", "-i", str(segment),
         "-frames:v", "1", "-q:v", "2", str(tmp)],
        capture_output=True, timeout=60,
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dest)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True, type=Path)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval-min", type=int, default=15)
    ap.add_argument("--oldest-first", action="store_true", default=True)
    a = ap.parse_args()

    days = sorted(p for p in a.recordings.iterdir() if p.is_dir())
    done = skipped = failed = 0
    for day in days:
        for hour in sorted(p for p in day.iterdir() if p.is_dir()):
            cam_dir = hour / a.camera
            if not cam_dir.is_dir():
                continue
            segs = segments_for_hour(cam_dir)
            if not segs:
                continue
            # Recording folders are UTC.
            base = dt.datetime.strptime(f"{day.name} {hour.name}", "%Y-%m-%d %H").replace(tzinfo=dt.timezone.utc)
            for minute in range(0, 60, a.interval_min):
                slot = minute * 60
                # The segment covering the slot, else (recordings outside continuous retention
                # are kept only around activity) the nearest segment within five minutes.
                covering = [(s, p) for s, p in segs if s <= slot and slot - s < 20]
                near = sorted(segs, key=lambda t: abs(t[0] - slot))
                if covering:
                    start, seg = covering[-1]
                elif near and abs(near[0][0] - slot) <= 300:
                    start, seg = near[0]
                else:
                    failed += 1
                    continue
                ts = base + dt.timedelta(seconds=slot)
                dest = raw_path(a.out, a.camera, ts)
                if dest.exists():
                    skipped += 1
                    continue
                if extract(seg, max(0.0, min(slot - start, 8.0)), dest):
                    done += 1
                else:
                    failed += 1
        print(f"{day.name}: done={done} skipped={skipped} missing={failed}", flush=True)


if __name__ == "__main__":
    main()
