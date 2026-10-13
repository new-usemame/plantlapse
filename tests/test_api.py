import pytest
import importlib


def make_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTLAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLANTLAPSE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PLANTLAPSE_TOKEN", "t0ken")
    monkeypatch.delenv("PLANTLAPSE_STREAM_URL", raising=False)
    (tmp_path / "state").mkdir()
    import plantlapse.api as api
    return importlib.reload(api)


def test_redact_strips_userinfo_from_urls():
    from plantlapse.api import redact
    s = "rtsp://admin:hunter2@192.0.2.5:554/ch0: 401 Unauthorized; http://u:p@h/x"
    out = redact(s)
    assert "hunter2" not in out and "u:p@" not in out and "rtsp://***@192.0.2.5:554/ch0" in out


def test_public_health_never_carries_error_text(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    api = make_app(tmp_path, monkeypatch)
    api._state["last_error"] = {"at": "20261012T000000Z", "detail": "rtsp://***@cam/x refused"}
    c = TestClient(api.app)
    h = c.get("/health").json()
    assert h["last_error"] == {"at": "20261012T000000Z", "failed": True}
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"Authorization": "Bearer t0ken"}).json()["last_error"]["detail"]


def test_plant_thumbnail_is_the_newest_located_daylight_frame(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    api = make_app(tmp_path, monkeypatch)
    from plantlapse import db as dbm
    c = dbm.connect(api.cfg.db_path)
    c.execute("INSERT INTO plants(id, name, created_ts) VALUES ('p1', 'Plant 1', '20261012T130000Z')")
    for ts, located, mode in [("20261012T140000Z", 1, "day"), ("20261012T141500Z", 0, "day"),
                              ("20261012T235000Z", 1, "night")]:
        c.execute("INSERT INTO frames(plant_id, ts, located, mode, crop) VALUES ('p1', ?, ?, ?, '[0,0,1,1]')",
                  (ts, located, mode))
    j = TestClient(api.app).get("/plants", headers={"Authorization": "Bearer t0ken"}).json()[0]
    assert j["thumbnail"] == "/plants/p1/frames/20261012T140000Z.jpg"
    assert j["thumbnail_frame"]["ts"] == "2026-10-12T14:00:00Z" and j["thumbnail_frame"]["located"] is True


@pytest.mark.parametrize("secret_url", [
    "rtsp://admin:p@ss@cam:554/ch0",     # "@" inside the password
    "rtsp://admin:pa/ss@cam:554/ch0",    # "/" inside the password
    "rtsp://admin:a/b@c/d@cam:554/ch0",  # both
])
def test_redact_handles_passwords_containing_at_and_slash(secret_url):
    from plantlapse.api import redact
    out = redact(f"[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized {secret_url}: error")
    assert "admin" not in out and "ss" not in out.replace("DESCRIBE", "") and "rtsp://***@cam:554/ch0" in out


def test_auto_fps_makes_short_histories_watchable_and_caps_long_ones(tmp_path, monkeypatch):
    api = make_app(tmp_path, monkeypatch)
    assert api.auto_fps(40) == 5          # 40 frames -> 8 s, not 1.7 s at 24 fps
    assert api.auto_fps(10) == 4
    assert api.auto_fps(330) == 24 and api.auto_fps(5000) == 24


def test_redact_strips_token_like_query_values():
    from plantlapse.api import redact
    out = redact("GET http://cam/x/ch0.jpg?token=s3cr3tT0k&w=1 failed; http://h/a?api_key=K9&password=p@ss")
    assert "s3cr3tT0k" not in out and "K9" not in out and "p@ss" not in out and "&w=1" in out
