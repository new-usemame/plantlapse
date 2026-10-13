import datetime as dt
import http.server
import io
import threading

import pytest
from PIL import Image

from plantlapse import tick
from plantlapse.config import Config


def _jpeg(color):
    b = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(b, "JPEG")
    return b.getvalue()


@pytest.fixture
def server():
    """A camera web server: /x/ch0.jpg is a JPEG; any other path answers 200 with an HTML page."""
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body, ctype = (_jpeg("green"), "image/jpeg") if self.path.startswith("/x/ch0.jpg") else (b"<html>recovery</html>", "text/html")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}", hits
    s.shutdown()


def cfg(tmp_path, snapshot_url):
    return Config(stream_url="rtsp://stream", camera="c", snapshot_url=snapshot_url, data_dir=tmp_path / "d",
                  state_dir=tmp_path / "s", interval_min=15, api_token=None, burst=3, crop_px=768, torch_threads=1)


def test_snapshots_are_used_when_they_are_jpegs(tmp_path, server, monkeypatch):
    base, hits = server
    monkeypatch.setattr(tick, "grab_burst", lambda *a: pytest.fail("stream used although snapshots worked"))
    monkeypatch.setattr(tick.time, "sleep", lambda s: None)
    ts, dest, _ = tick.capture(cfg(tmp_path, f"{base}/x/ch0.jpg?token=T"))
    assert len(hits) == 3 and Image.open(dest).getpixel((10, 10))[1] > 100


def test_an_html_page_answering_200_is_not_a_snapshot_and_the_stream_takes_over(tmp_path, server, monkeypatch):
    base, _ = server
    used = []

    def burst(url, n, workdir):
        used.append(url)
        f = workdir / "f01.jpg"
        f.write_bytes(_jpeg("red"))
        return [f]

    monkeypatch.setattr(tick, "grab_burst", burst)
    ts, dest, _ = tick.capture(cfg(tmp_path, f"{base}/wrong/path.jpg?token=T"))
    assert used == ["rtsp://stream"] and Image.open(dest).getpixel((10, 10))[0] > 100


def test_a_failed_snapshot_error_never_carries_the_url(tmp_path):
    with pytest.raises(RuntimeError) as e:
        tick.grab_snapshots("http://127.0.0.1:9/x/ch0.jpg?token=SECRETVALUE", 1, tmp_path)
    assert "SECRETVALUE" not in str(e.value) and "127.0.0.1" not in str(e.value)


def test_reframe_recuts_only_the_zoomed_crops_and_keeps_the_identity(tmp_path):
    import json
    from plantlapse import db as dbm
    from plantlapse.storage import crop_path, raw_path, parse_stamp

    c = cfg(tmp_path, None)
    conn = dbm.connect(c.db_path)
    conn.execute("INSERT INTO plants(id, name, created_ts, size, anchor_x, anchor_y) VALUES ('pX','Plant 1','t',373,1290,678)")
    # The carnivorous plant's real boxes; the crops stored at 15:45 and 16:00 were zoomed out by the old rule.
    real = [(1221, 580, 1366, 759), (1218, 593, 1359, 758), (1216, 602, 1356, 758), (1219, 550, 1380, 758),
            (1218, 519, 1402, 759), (1221, 598, 1362, 758), (1218, 606, 1336, 757)]
    stamps = ["20261012T150000Z", "20261012T151500Z", "20261012T153000Z", "20261012T154500Z",
              "20261012T160000Z", "20261012T161500Z", "20261012T163000Z"]
    old_crops = [[1152, 554, 1400, 802]] * 4 + [[1089, 492, 1462, 865]] * 2 + [[1152, 554, 1400, 802]]
    for st, b, cr in zip(stamps, real, old_crops):
        r = raw_path(c.data_dir, c.camera, parse_stamp(st))
        r.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), "gray").save(r)
        out = crop_path(c.data_dir, "pX", parse_stamp(st))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"old")
        conn.execute("INSERT INTO frames(plant_id, ts, located, mode, box, crop, sim) VALUES ('pX',?,1,'day',?,?,0.88)",
                     (st, json.dumps(b), json.dumps(cr)))
    assert tick.cmd_reframe(c) == 0
    crops = [json.loads(r[0]) for r in conn.execute("SELECT crop FROM frames WHERE plant_id='pX' ORDER BY ts")]
    assert len({tuple(x) for x in crops}) == 1, crops
    rewritten = [crop_path(c.data_dir, "pX", parse_stamp(st)).read_bytes() != b"old" for st in stamps]
    assert rewritten[4] and rewritten[5]  # the two zoomed-out crops were re-cut from their raw frames
    assert conn.execute("SELECT id, name FROM plants").fetchall()[0][:] == ("pX", "Plant 1")


def test_reframe_cuts_a_stretch_at_the_size_it_settled_on_so_the_history_never_zooms(tmp_path):
    import json
    from plantlapse import db as dbm
    from plantlapse.storage import raw_path, parse_stamp

    c = cfg(tmp_path, None)
    conn = dbm.connect(c.db_path)
    conn.execute("INSERT INTO plants(id, name, created_ts) VALUES ('pB','Plant 1','t')")
    # The basil's real boxes: short (top ~616) for the first hours, then mostly tall (top ~576).
    short, tall = (585, 616, 687, 732), (598, 578, 679, 731)
    seq = [short] * 6 + [tall] * 6
    for i, b in enumerate(seq):
        st = f"20261012T{13 + i // 4:02d}{(i % 4) * 15:02d}00Z"
        r = raw_path(c.data_dir, c.camera, parse_stamp(st))
        r.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), "gray").save(r)
        conn.execute("INSERT INTO frames(plant_id, ts, located, mode, box, crop, sim) VALUES ('pB',?,1,'day',?,'[0,0,1,1]',0.88)",
                     (st, json.dumps(b)))
    tick.cmd_reframe(c)
    sides = {json.loads(r[0])[2] - json.loads(r[0])[0] for r in conn.execute("SELECT crop FROM frames")}
    assert len(sides) == 1 and sides.pop() > 1.18 * 1.55 * 116
