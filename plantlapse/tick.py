"""One capture slot: grab a frame, find every plant in it, and write one framed picture per plant.

    python -m plantlapse.tick capture            # live: grab now, then process it
    python -m plantlapse.tick process [--reset] [--since STAMP]  # (re)process stored raw frames not yet processed
    python -m plantlapse.tick reframe            # re-cut every plant's crops under the current framing rules

Runs as a short-lived process so the models' memory is returned to the host between slots.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

from . import config as config_mod
from . import db as dbm
from .storage import crop_path, parse_stamp, raw_path, stamp
from .tracker import Candidate, Det, Params, Plant, crop_window, plant_from_candidate, step, update_framing


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def slot_time(now: dt.datetime, interval_min: int) -> dt.datetime:
    u = now.astimezone(dt.timezone.utc)
    m = (u.minute // interval_min) * interval_min
    return u.replace(minute=m, second=0, microsecond=0)


def grab_burst(url: str, n: int, workdir: Path) -> list[Path]:
    args = ["ffmpeg", "-loglevel", "error", "-y"]
    if url.startswith("rtsp"):
        args += ["-rtsp_transport", "tcp"]
    # Keyframes only: a grab that starts between keyframes decodes smeared, half-referenced
    # pictures ("Missing reference picture"). Typical IP cameras send one every 1–4 s.
    args += ["-skip_frame", "nokey", "-i", url, "-frames:v", str(n), "-fps_mode", "passthrough",
             "-q:v", "2", str(workdir / "f%02d.jpg")]
    subprocess.run(args, check=True, capture_output=True, timeout=60)
    return sorted(workdir.glob("f*.jpg"))


def grab_snapshots(url: str, n: int, workdir: Path, gap_s: float = 1.5) -> list[Path]:
    """n stills from a snapshot URL, gap_s apart. Only real JPEGs count: some camera web servers answer
    a wrong path with 200 and an HTML page. Errors never carry the URL, which may hold a token."""
    import urllib.request

    out: list[Path] = []
    for i in range(n):
        if i:
            time.sleep(gap_s)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                body = r.read(20_000_000)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"snapshot request failed: {type(e).__name__} {getattr(e, 'code', '')}".rstrip()) from None
        if not body.startswith(b"\xff\xd8"):
            raise RuntimeError(f"snapshot is not a JPEG ({len(body)} bytes)")
        f = workdir / f"s{i:02d}.jpg"
        f.write_bytes(body)
        Image.open(f).load()
        out.append(f)
    return out


def capture(cfg: config_mod.Config) -> tuple[dt.datetime, Path, float]:
    from .vision import sharpness

    ts = slot_time(dt.datetime.now(dt.timezone.utc), cfg.interval_min)
    dest = raw_path(cfg.data_dir, cfg.camera, ts)
    with tempfile.TemporaryDirectory() as td:
        frames: list[Path] = []
        if cfg.snapshot_url:
            try:
                frames = grab_snapshots(cfg.snapshot_url, cfg.burst, Path(td))
            except Exception as e:  # noqa: BLE001 — the stream is the fallback
                log(f"snapshot failed, using the stream: {e}")
        last: Exception | None = None
        for attempt in range(0 if frames else 3):
            try:
                frames = grab_burst(cfg.stream_url, cfg.burst, Path(td))
                if frames:
                    break
            except Exception as e:  # noqa: BLE001 — retried, then reported
                last = e
                time.sleep(5)
        if not frames:
            raise RuntimeError(f"no frames from stream after 3 attempts: {last}")
        # Wind blurs leaves; keep the sharpest frame of the burst.
        scored = [(sharpness(Image.open(f)), f) for f in frames]
        best_s, best = max(scored)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        shutil.copyfile(best, tmp)
        os.replace(tmp, dest)
    return ts, dest, best_s


# ---------------------------------------------------------------- persistence


def load_plants(c) -> list[Plant]:
    out = []
    for r in c.execute("SELECT * FROM plants WHERE merged_into IS NULL ORDER BY created_ts"):
        pl = Plant(
            id=r["id"],
            anchor=(r["anchor_x"], r["anchor_y"]) if r["anchor_x"] is not None else None,
            size=r["size"],
            last_box=tuple(json.loads(r["last_box"])) if r["last_box"] else None,
            miss_streak=r["miss_streak"],
            displaced_streak=r["displaced_streak"],
            sizes=json.loads(r["sizes"]),
            score=r["score"],
        )
        pl.protos = [dbm.blob_to_vec(x["vec"]) for x in
                     c.execute("SELECT vec FROM protos WHERE plant_id=? AND mode='day' ORDER BY rowid", (pl.id,))]
        out.append(pl)
    return out


def load_candidates(c) -> list[Candidate]:
    return [Candidate(r["id"], tuple(json.loads(r["box"])), dbm.blob_to_vec(r["vec"]), r["hits"],
                      r["first_ts"], r["last_ts"], r["stale"], [tuple(b) for b in json.loads(r["boxes"])],
                      None, json.loads(r["old_sims"]))
            for r in c.execute("SELECT * FROM candidates ORDER BY id")]


def save_plant(c, pl: Plant, ts: str, located: bool) -> None:
    c.execute(
        """UPDATE plants SET anchor_x=?, anchor_y=?, size=?, last_box=?, miss_streak=?,
           displaced_streak=?, sizes=?, score=?, last_located_ts=COALESCE(?, last_located_ts) WHERE id=?""",
        (pl.anchor[0] if pl.anchor else None, pl.anchor[1] if pl.anchor else None, pl.size,
         dbm.j(list(pl.last_box)) if pl.last_box else None, pl.miss_streak, pl.displaced_streak,
         dbm.j(pl.sizes), pl.score, ts if located else None, pl.id))
    for v in pl.new_protos:
        c.execute("INSERT INTO protos(plant_id, mode, ts, vec) VALUES (?,?,?,?)", (pl.id, "day", ts, dbm.vec_to_blob(v)))
    pl.new_protos = []
    # Mirror the in-memory pruning so the table cannot grow without bound.
    rows = [r[0] for r in c.execute("SELECT rowid FROM protos WHERE plant_id=? AND mode='day' ORDER BY rowid", (pl.id,))]
    if len(rows) > Params().max_protos:
        drop = rows[8:len(rows) - (Params().max_protos - 8)]
        c.executemany("DELETE FROM protos WHERE rowid=?", [(r,) for r in drop])


def new_plant_id(c) -> tuple[str, str]:
    n = c.execute("SELECT COUNT(*) FROM plants").fetchone()[0] + 1
    return "p" + secrets.token_hex(3), f"Plant {n}"


# ---------------------------------------------------------------- processing


def day_before(cfg: config_mod.Config, ts_dt: dt.datetime) -> Image.Image | None:
    """The stored frame from the same time of day, one to three days earlier (same light), if any."""
    for days in (1, 2, 3):
        for minutes in (0, -15, 15, -30, 30):
            p = raw_path(cfg.data_dir, cfg.camera, ts_dt - dt.timedelta(days=days, minutes=minutes))
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:  # noqa: BLE001 — a damaged old frame just means no comparison
                    continue
    return None


def process_frame(c, models, cfg: config_mod.Config, ts_dt: dt.datetime, img: Image.Image,
                  sharp: float | None = None) -> dict:
    from .vision import lighting_mode

    ts = stamp(ts_dt)
    img = img.convert("RGB")
    W, H = img.size
    mode = lighting_mode(img, ts_dt, cfg.latlon)
    plants = load_plants(c)
    focus = [pl.last_box for pl in plants if pl.last_box]
    # Night frames are framed at each plant's last daylight position without running the
    # models: identity is only ever decided in daylight, so detection at night buys nothing.
    found = models.detect(img, focus=focus) if mode == "day" else []
    vecs = models.embed(img, [b for b, _, _ in found])
    dets = [Det(b, s, v, kind) for (b, s, kind), v in zip(found, vecs)]
    plant_dets = [d for d in dets if d.kind == "plant"]
    old = day_before(cfg, ts_dt) if plant_dets else None
    if old is not None:
        for d, ov in zip(plant_dets, models.embed(old, [d.box for d in plant_dets])):
            d.old_sim = float(d.vec @ ov)
    cands = load_candidates(c)

    res = step(plants, cands, dets, mode, (W, H), ts)

    c.execute("BEGIN")
    try:
        for pl in plants:
            save_plant(c, pl, ts, pl.id in res.matches)
        c.execute("DELETE FROM candidates")
        for cd in res.candidates:
            c.execute("INSERT INTO candidates(box, vec, hits, first_ts, last_ts, stale, boxes, old_sims) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (dbm.j(list(cd.box)), dbm.vec_to_blob(cd.vec), cd.hits, cd.first_ts, cd.last_ts, cd.stale,
                       dbm.j([list(b) for b in cd.boxes[-10:]]), dbm.j(cd.old_sims[-10:])))
        for cd in res.promoted:
            pid, name = new_plant_id(c)
            pl = plant_from_candidate(pid, cd)
            # Scenery is still tracked, so it keeps claiming its own box and is never founded again,
            # but hidden: the app shows it only under "hidden", where one tap brings it back.
            if cd.scenery:
                name = "Was already there " + name.split()[-1]
            c.execute("INSERT INTO plants(id, name, created_ts, hidden) VALUES (?,?,?,?)",
                      (pid, name, ts, int(cd.scenery)))
            save_plant(c, pl, ts, True)
            plants.append(pl)
            res.matches[pid] = None  # type: ignore[assignment]
            log(f"new {'SCENERY (hidden)' if cd.scenery else 'plant'} {pid} ({name}) at "
                f"{[round(v) for v in cd.box]} after {cd.hits} sightings, day-before likeness {cd.old_sims}")

        for pl in plants:
            if pl.anchor is None or pl.size is None:
                continue
            win = crop_window(pl, (W, H))
            out = crop_path(cfg.data_dir, pl.id, ts_dt)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            img.crop(win).resize((cfg.crop_px, cfg.crop_px), Image.LANCZOS).save(tmp, "JPEG", quality=92)
            os.replace(tmp, out)
            m = res.matches.get(pl.id)
            located = pl.id in res.matches
            det = dets[m.det] if m is not None else None
            c.execute(
                "INSERT OR REPLACE INTO frames(plant_id, ts, located, mode, box, crop, score, sim) VALUES (?,?,?,?,?,?,?,?)",
                (pl.id, ts, int(located), mode, dbm.j(list(det.box)) if det else (dbm.j(list(pl.last_box)) if located and pl.last_box else None),
                 dbm.j(list(win)), det.score if det else None, m.sim if m is not None else None))
        c.execute(
            "INSERT OR REPLACE INTO captures(ts, camera, mode, sharpness, dets, processed) VALUES (?,?,?,?,?,1)",
            (ts, cfg.camera, mode, sharp, dbm.j([[list(d.box), round(d.score, 3), d.kind] for d in dets])))
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    return {"ts": ts, "mode": mode, "dets": len(dets), "plants": len(plants),
            "located": sum(1 for pl in plants if pl.id in res.matches)}


def cmd_capture(cfg: config_mod.Config) -> int:
    from .vision import Models

    ts, path, sharp = capture(cfg)
    log(f"captured {path} sharpness={sharp:.0f}")
    c = dbm.connect(cfg.db_path)
    models = Models(cfg.torch_threads)
    r = process_frame(c, models, cfg, ts, Image.open(path), sharp)
    log(f"processed {r}")
    c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('last_ok', ?)", (r["ts"],))
    return 0


def cmd_process(cfg: config_mod.Config, reset: bool, since: str | None = None) -> int:
    from .vision import Models, sharpness

    if reset:
        if cfg.db_path.exists():
            for suffix in ("", "-wal", "-shm"):
                Path(str(cfg.db_path) + suffix).unlink(missing_ok=True)
        shutil.rmtree(cfg.data_dir / "plants", ignore_errors=True)
    c = dbm.connect(cfg.db_path)
    done = {r[0] for r in c.execute("SELECT ts FROM captures WHERE processed=1")}
    raws = sorted((cfg.data_dir / "raw" / cfg.camera).rglob("*.jpg"))
    todo = [p for p in raws if p.stem not in done and not p.name.endswith(".part.jpg")
            and (since is None or p.stem >= since)]
    log(f"{len(todo)} raw frames to process ({len(done)} already done)")
    models = Models(cfg.torch_threads)
    t0 = time.time()
    for i, p in enumerate(todo):
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # noqa: BLE001
            log(f"skip unreadable {p}: {e}")
            continue
        r = process_frame(c, models, cfg, parse_stamp(p.stem), img, sharpness(img))
        if i % 25 == 0:
            log(f"[{i + 1}/{len(todo)}] {r} {(time.time() - t0) / (i + 1):.1f}s/frame")
    return 0


def cmd_reframe(cfg: config_mod.Config) -> int:
    """Replay each plant's framing from its stored boxes and re-cut the crops whose window changed.
    Identities, names and IDs are untouched; only framing (anchor, size) and crop files change.

    Unlike a live capture, a replay knows what came after, so each stretch between moves of the plant is
    cut at the size that stretch settled on: the timelapse never zooms while the plant stays put."""
    c = dbm.connect(cfg.db_path)
    p = Params()
    for pl in load_plants(c):
        fresh = Plant(id=pl.id)
        rows = c.execute("SELECT ts, located, mode, box, crop, sim FROM frames WHERE plant_id=? ORDER BY ts",
                         (pl.id,)).fetchall()
        states: list[tuple[int, tuple[float, float], float] | None] = []  # (stretch, anchor, size) per row
        stretch = 0
        for r in rows:
            if r["located"] and r["mode"] == "day" and r["box"]:
                had = fresh.anchor is not None
                update_framing(fresh, tuple(json.loads(r["box"])), r["sim"], p)
                if had and fresh.sizes == [fresh.size]:  # the plant moved: a new stretch begins
                    stretch += 1
            states.append((stretch, fresh.anchor, fresh.size) if fresh.anchor is not None else None)
        settled = {st[0]: st[2] for st in states if st}  # the last size of each stretch
        changed = missing = 0
        for r, st in zip(rows, states):
            if st is None:
                continue
            fresh.anchor, fresh.size = st[1], settled[st[0]]
            ts_dt = parse_stamp(r["ts"])
            raw = raw_path(cfg.data_dir, cfg.camera, ts_dt)
            if not raw.exists():
                missing += 1
                continue
            with Image.open(raw) as img:
                win = crop_window(fresh, img.size)
                if list(win) == json.loads(r["crop"]):
                    continue
                out = crop_path(cfg.data_dir, pl.id, ts_dt)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".part")
                img.convert("RGB").crop(win).resize((cfg.crop_px, cfg.crop_px), Image.LANCZOS).save(tmp, "JPEG", quality=92)
                os.replace(tmp, out)
            c.execute("UPDATE frames SET crop=? WHERE plant_id=? AND ts=?", (dbm.j(list(win)), pl.id, r["ts"]))
            changed += 1
        if fresh.anchor is not None:
            c.execute("UPDATE plants SET anchor_x=?, anchor_y=?, size=?, sizes=? WHERE id=?",
                      (fresh.anchor[0], fresh.anchor[1], fresh.size, dbm.j(fresh.sizes), pl.id))
        log(f"reframe {pl.id}: {changed} of {len(rows)} crops re-cut, {missing} without a raw frame")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    sub.add_parser("reframe")
    pp = sub.add_parser("process")
    pp.add_argument("--reset", action="store_true")
    pp.add_argument("--since", help="only raws at or after this stamp, e.g. 20261011T000000Z (older ones still "
                                    "serve as the day-before comparison for scenery)")
    a = ap.parse_args()
    cfg = config_mod.load()
    if a.cmd == "capture":
        return cmd_capture(cfg)
    if a.cmd == "reframe":
        return cmd_reframe(cfg)
    return cmd_process(cfg, a.reset, a.since)


if __name__ == "__main__":
    sys.exit(main())
