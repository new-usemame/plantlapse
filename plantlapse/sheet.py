"""Review aids: a contact sheet per plant, and whole frames with every plant's box drawn.

    python -m plantlapse.sheet OUT_DIR [--every N] [--since STAMP]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from . import config as config_mod
from . import db as dbm
from .storage import crop_path, parse_stamp, raw_path

COLOURS = [(255, 214, 0), (0, 200, 255), (255, 80, 160), (120, 255, 120), (255, 140, 0), (180, 120, 255)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--since", default="")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args()
    cfg = config_mod.load()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = dbm.connect(cfg.db_path)
    plants = c.execute("SELECT * FROM plants WHERE merged_into IS NULL ORDER BY created_ts").fetchall()
    for r in plants:
        rows = c.execute("SELECT * FROM frames WHERE plant_id=? AND mode='day' AND ts>=? ORDER BY ts",
                         (r["id"], a.since)).fetchall()[::a.every]
        if not rows:
            continue
        t = 192
        n = len(rows)
        cols = min(a.cols, n)
        sheet = Image.new("RGB", (cols * t, ((n + cols - 1) // cols) * (t + 14)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(rows):
            p = crop_path(cfg.data_dir, r["id"], parse_stamp(f["ts"]))
            if not p.exists():
                continue
            x, y = (i % cols) * t, (i // cols) * (t + 14)
            sheet.paste(Image.open(p).resize((t, t)), (x, y))
            col = (255, 255, 255) if f["located"] else (255, 60, 60)
            d.text((x + 2, y + t), f["ts"][4:13] + ("" if f["located"] else " MISS")
                   + (f" {f['sim']:.2f}" if f["sim"] is not None else ""), fill=col)
        sheet.save(out / f"sheet_{r['name'].replace(' ', '_')}_{r['id']}.jpg", quality=85)
    # Annotated whole frames: a handful across the range.
    caps = c.execute("SELECT ts FROM captures WHERE mode='day' AND ts>=? ORDER BY ts", (a.since,)).fetchall()
    for cap in caps[:: max(1, len(caps) // 6)]:
        ts = cap["ts"]
        img = Image.open(raw_path(cfg.data_dir, cfg.camera, parse_stamp(ts))).convert("RGB")
        d = ImageDraw.Draw(img)
        dets = json.loads(c.execute("SELECT dets FROM captures WHERE ts=?", (ts,)).fetchone()[0] or "[]")
        for b, s, *k in dets:
            d.rectangle(b, outline=(90, 90, 90) if (k and k[0] == "proposal") else (200, 200, 200), width=1)
        for i, r in enumerate(plants):
            f = c.execute("SELECT * FROM frames WHERE plant_id=? AND ts=?", (r["id"], ts)).fetchone()
            if not f:
                continue
            col = COLOURS[i % len(COLOURS)]
            d.rectangle(json.loads(f["crop"]), outline=col, width=3)
            if f["box"] and f["located"]:
                d.rectangle(json.loads(f["box"]), outline=col, width=1)
            x0, y0 = json.loads(f["crop"])[:2]
            d.text((x0 + 6, y0 + 4), r["name"] + ("" if f["located"] else " (not found)"), fill=col)
        img.save(out / f"frame_{ts}.jpg", quality=80)
    print(f"{len(plants)} plants:", ", ".join(f"{r['name']}={r['id']}" for r in plants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
