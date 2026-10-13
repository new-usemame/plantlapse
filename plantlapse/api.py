"""HTTP API and the capture scheduler.

Every route except /health needs `Authorization: Bearer <PLANTLAPSE_TOKEN>` when a token is set.
The scheduler runs the capture as a child process on each slot, so the models never stay resident.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import config as config_mod
from . import db as dbm
from .render import render
from .storage import crop_path, parse_stamp, raw_path, stamp

cfg = config_mod.load()
app = FastAPI(title="plantlapse")
_state = {"last_run": None, "last_error": None, "running": False}
_render_lock = threading.Lock()

NOT_SEEN_AFTER = 8  # daylight captures without a sighting before a plant reads "not seen"
RANGES = {"day": 1, "week": 7, "month": 31, "all": None}
# Greedy to the LAST "@" in the URL token: a password may itself contain "@" or "/", and stopping at
# the first of either leaked the rest of it. Over-redacting a path that contains "@" is harmless.
_URL_USERINFO = re.compile(r"(\w+://)\S*@")
_URL_SECRET_PARAM = re.compile(r"([?&](?:token|access_token|api_?key|key|password|passwd|pwd|pass|auth|secret|sig)=)[^&\s]+", re.I)


def redact(text: str) -> str:
    """Strip user:password and token-like query values from any URL; ffmpeg echoes its input URL in errors."""
    return _URL_SECRET_PARAM.sub(r"\1***", _URL_USERINFO.sub(r"\1***@", text))


def conn():
    return dbm.connect(cfg.db_path)


def auth(request: Request) -> None:
    if not cfg.api_token:
        return
    h = request.headers.get("authorization", "")
    if h != f"Bearer {cfg.api_token}":
        raise HTTPException(401, "missing or wrong bearer token")


# ---------------------------------------------------------------- scheduler


def _next_slot(now: dt.datetime) -> dt.datetime:
    step = cfg.interval_min
    base = now.replace(second=0, microsecond=0)
    m = (base.minute // step + 1) * step
    return base.replace(minute=0) + dt.timedelta(minutes=m)


def _scheduler() -> None:
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        nxt = _next_slot(now) + dt.timedelta(seconds=5)
        time.sleep(max(1.0, (nxt - now).total_seconds()))
        now = dt.datetime.now(dt.timezone.utc)
        _state["running"] = True
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-m", "plantlapse.tick", "capture"],
                               capture_output=True, text=True, timeout=cfg.interval_min * 60 - 30)
            sys.stdout.write(redact(r.stdout[-4000:]))
            sys.stdout.write(redact("".join(line + "\n" for line in r.stderr.splitlines() if "Warning" not in line)[-4000:]))
            sys.stdout.flush()
            if r.returncode != 0:
                _state["last_error"] = {"at": stamp(now), "detail": redact((r.stderr or r.stdout)[-500:])}
            else:
                _state["last_error"] = None
        except Exception as e:  # noqa: BLE001
            _state["last_error"] = {"at": stamp(now), "detail": redact(str(e)[-500:])}
        finally:
            _state["running"] = False
            _state["last_run"] = {"at": stamp(now), "seconds": round(time.time() - t0, 1)}


@app.on_event("startup")
def _start() -> None:
    if cfg.stream_url:
        threading.Thread(target=_scheduler, daemon=True).start()


# ---------------------------------------------------------------- helpers


def _plant_row(c, pid: str):
    r = c.execute("SELECT * FROM plants WHERE id=?", (pid,)).fetchone()
    if r is None:
        raise HTTPException(404, "no such plant")
    if r["merged_into"]:
        return _plant_row(c, r["merged_into"])
    return r


def _plant_json(c, r) -> dict:
    stats = c.execute(
        """SELECT COUNT(*) n, SUM(mode='day') nd, MIN(ts) first, MAX(ts) last,
                  MAX(CASE WHEN mode='day' THEN ts END) last_day
           FROM frames WHERE plant_id=?""", (r["id"],)).fetchone()
    status = "tracking" if r["miss_streak"] < NOT_SEEN_AFTER else "not_seen"
    # The picture on the plant's card: its newest daylight frame in which it was actually found.
    thumb = c.execute("""SELECT ts, located, mode FROM frames WHERE plant_id=? AND mode='day' AND located=1
                         ORDER BY ts DESC LIMIT 1""", (r["id"],)).fetchone()
    return {
        "id": r["id"],
        "name": r["name"],
        "hidden": bool(r["hidden"]),
        "status": status,
        "created": _iso(r["created_ts"]),
        "last_seen": _iso(r["last_located_ts"]),
        "frames": stats["n"] or 0,
        "daylight_frames": stats["nd"] or 0,
        "first_frame": _iso(stats["first"]),
        "last_frame": _iso(stats["last"]),
        "thumbnail": f"/plants/{r['id']}/frames/{thumb['ts']}.jpg" if thumb else None,
        "thumbnail_frame": {"ts": _iso(thumb["ts"]), "stamp": thumb["ts"], "located": True, "mode": thumb["mode"]}
        if thumb else None,
    }


def _iso(s: str | None) -> str | None:
    return parse_stamp(s).isoformat().replace("+00:00", "Z") if s else None


def _frames(c, pid: str, daylight: bool, since: str | None = None, before: str | None = None,
            limit: int | None = None, desc: bool = False) -> list:
    q = "SELECT * FROM frames WHERE plant_id=?"
    a: list = [pid]
    if daylight:
        q += " AND mode='day'"
    if since:
        q += " AND ts>=?"
        a.append(since)
    if before:
        q += " AND ts<?"
        a.append(before)
    q += " ORDER BY ts " + ("DESC" if desc else "ASC")
    if limit:
        q += " LIMIT ?"
        a.append(limit)
    return c.execute(q, a).fetchall()


# ---------------------------------------------------------------- routes


@app.get("/health")
def health() -> dict:
    c = conn()
    last = c.execute("SELECT MAX(ts) FROM captures WHERE processed=1").fetchone()[0]
    age = (dt.datetime.now(dt.timezone.utc) - parse_stamp(last)).total_seconds() if last else None
    late = age is None or age > cfg.interval_min * 60 * 2 + 120
    return {
        "ok": not late and _state["last_error"] is None,
        "camera": cfg.camera,
        "interval_min": cfg.interval_min,
        "last_capture": _iso(last),
        "last_capture_age_s": round(age) if age is not None else None,
        "last_run": _state["last_run"],
        # Public route: whether the last run failed, never its text. Detail is behind the token at /status.
        "last_error": {"at": _state["last_error"]["at"], "failed": True} if _state["last_error"] else None,
        "running": _state["running"],
        "plants": c.execute("SELECT COUNT(*) FROM plants WHERE merged_into IS NULL AND hidden=0").fetchone()[0],
    }


@app.get("/status", dependencies=[Depends(auth)])
def status() -> dict:
    return {**health(), "last_error": _state["last_error"]}


@app.get("/plants", dependencies=[Depends(auth)])
def plants(include_hidden: bool = False) -> list[dict]:
    c = conn()
    q = "SELECT * FROM plants WHERE merged_into IS NULL" + ("" if include_hidden else " AND hidden=0")
    return [_plant_json(c, r) for r in c.execute(q + " ORDER BY created_ts")]


@app.get("/plants/{pid}", dependencies=[Depends(auth)])
def plant(pid: str) -> dict:
    c = conn()
    return _plant_json(c, _plant_row(c, pid))


class PlantPatch(BaseModel):
    name: str | None = None
    hidden: bool | None = None


@app.patch("/plants/{pid}", dependencies=[Depends(auth)])
def patch_plant(pid: str, body: PlantPatch) -> dict:
    c = conn()
    r = _plant_row(c, pid)
    if body.name is not None:
        name = body.name.strip()[:60]
        if not name:
            raise HTTPException(422, "name must not be empty")
        c.execute("UPDATE plants SET name=? WHERE id=?", (name, r["id"]))
    if body.hidden is not None:
        c.execute("UPDATE plants SET hidden=? WHERE id=?", (int(body.hidden), r["id"]))
    return _plant_json(c, _plant_row(c, r["id"]))


class Merge(BaseModel):
    into: str


@app.post("/plants/{pid}/merge", dependencies=[Depends(auth)])
def merge(pid: str, body: Merge) -> dict:
    """Two identities that are really one plant: keep `into`, fold `pid`'s memory and frames into it."""
    c = conn()
    src, dst = _plant_row(c, pid), _plant_row(c, body.into)
    if src["id"] == dst["id"]:
        raise HTTPException(422, "cannot merge a plant into itself")
    c.execute("BEGIN")
    c.execute("UPDATE protos SET plant_id=? WHERE plant_id=?", (dst["id"], src["id"]))
    c.execute("""UPDATE frames SET plant_id=? WHERE plant_id=? AND ts NOT IN
                 (SELECT ts FROM frames WHERE plant_id=?)""", (dst["id"], src["id"], dst["id"]))
    c.execute("DELETE FROM frames WHERE plant_id=?", (src["id"],))
    c.execute("UPDATE plants SET merged_into=? WHERE id=?", (dst["id"], src["id"]))
    c.execute("COMMIT")
    return _plant_json(c, _plant_row(c, dst["id"]))


@app.get("/plants/{pid}/frames", dependencies=[Depends(auth)])
def frames(pid: str, limit: int = Query(200, le=5000), before: str | None = None,
           daylight: bool = True) -> list[dict]:
    c = conn()
    r = _plant_row(c, pid)
    out = []
    for f in _frames(c, r["id"], daylight, before=before, limit=limit, desc=True):
        out.append({"ts": _iso(f["ts"]), "stamp": f["ts"], "located": bool(f["located"]), "mode": f["mode"],
                    "url": f"/plants/{r['id']}/frames/{f['ts']}.jpg"})
    return out


def _frame_file(pid: str, s: str) -> Path:
    # A merged plant's frames keep their original path; look in both places.
    p = crop_path(cfg.data_dir, pid, parse_stamp(s))
    if p.exists():
        return p
    c = conn()
    for r in c.execute("SELECT id FROM plants WHERE merged_into=?", (pid,)):
        q = crop_path(cfg.data_dir, r["id"], parse_stamp(s))
        if q.exists():
            return q
    raise HTTPException(404, "no such frame")


@app.get("/plants/{pid}/frames/{s}.jpg", dependencies=[Depends(auth)])
def frame(pid: str, s: str) -> FileResponse:
    try:
        parse_stamp(s)
    except ValueError:
        raise HTTPException(404, "bad stamp") from None
    return FileResponse(_frame_file(_plant_row(conn(), pid)["id"], s), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


def auto_fps(n: int) -> int:
    """Frames per second that make a clip worth watching: about 8 s while history is short, up to 24 fps
    (a week of daylight, ~330 frames, is then ~14 s)."""
    return max(4, min(24, round(n / 8)))


@app.get("/plants/{pid}/timelapse.mp4", dependencies=[Depends(auth)])
def timelapse(pid: str, range: str = "week", daylight: bool = True, fps: int | None = Query(None, ge=2, le=60),
              size: int = Query(1080, ge=240, le=2160)) -> FileResponse:
    if range not in RANGES:
        raise HTTPException(422, f"range must be one of {list(RANGES)}")
    c = conn()
    r = _plant_row(c, pid)
    days = RANGES[range]
    since = stamp(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)) if days else None
    rows = _frames(c, r["id"], daylight, since=since)
    files = []
    for f in rows:
        try:
            files.append(_frame_file(r["id"], f["ts"]))
        except HTTPException:
            continue
    if len(files) < 2:
        raise HTTPException(409, "not enough frames yet for a timelapse")
    if fps is None:
        fps = auto_fps(len(files))
    with _render_lock:
        out = render(files, cfg.data_dir / "renders", fps=fps, size=size)
    name = f"{r['name']} {range}.mp4".replace("/", "-")
    return FileResponse(out, media_type="video/mp4", filename=name)


@app.get("/captures/latest.jpg", dependencies=[Depends(auth)])
def latest_capture(annotate: bool = True) -> Response:
    """The newest whole camera frame, with every plant's box and name drawn on it."""
    from PIL import Image, ImageDraw

    c = conn()
    cap = c.execute("SELECT * FROM captures WHERE processed=1 ORDER BY ts DESC LIMIT 1").fetchone()
    if cap is None:
        raise HTTPException(404, "no captures yet")
    img = Image.open(raw_path(cfg.data_dir, cfg.camera, parse_stamp(cap["ts"]))).convert("RGB")
    if annotate:
        d = ImageDraw.Draw(img)
        for f in c.execute("""SELECT f.*, p.name FROM frames f JOIN plants p ON p.id=f.plant_id
                              WHERE f.ts=? AND p.hidden=0""", (cap["ts"],)):
            x0, y0, x1, y1 = json.loads(f["crop"])
            d.rectangle((x0, y0, x1, y1), outline=(255, 214, 0), width=4)
            d.text((x0 + 8, y0 + 6), f["name"], fill=(255, 214, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-Capture-Time": _iso(cap["ts"]) or "", "Cache-Control": "no-store"})


@app.exception_handler(FileNotFoundError)
def _missing(_req, _exc) -> JSONResponse:
    return JSONResponse({"detail": "file missing"}, status_code=404)
