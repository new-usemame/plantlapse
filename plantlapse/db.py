"""SQLite state: plants, their appearance memory, candidates, captures and per-plant frames."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    hidden INTEGER NOT NULL DEFAULT 0,
    created_ts TEXT NOT NULL,
    anchor_x REAL, anchor_y REAL, size REAL,
    last_box TEXT,
    last_located_ts TEXT,
    miss_streak INTEGER NOT NULL DEFAULT 0,
    displaced_streak INTEGER NOT NULL DEFAULT 0,
    sizes TEXT NOT NULL DEFAULT '[]',
    merged_into TEXT
);
CREATE TABLE IF NOT EXISTS protos (
    plant_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    ts TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS protos_plant ON protos(plant_id, mode);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    box TEXT NOT NULL,
    vec BLOB NOT NULL,
    hits INTEGER NOT NULL,
    first_ts TEXT NOT NULL,
    last_ts TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    boxes TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS captures (
    ts TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    mode TEXT,
    sharpness REAL,
    dets TEXT,
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS frames (
    plant_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    located INTEGER NOT NULL,
    mode TEXT NOT NULL,
    box TEXT,
    crop TEXT NOT NULL,
    score REAL,
    sim REAL,
    PRIMARY KEY (plant_id, ts)
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    c.executescript(SCHEMA)
    cols = {r[1] for r in c.execute("PRAGMA table_info(plants)")}
    if "score" not in cols:
        c.execute("ALTER TABLE plants ADD COLUMN score REAL")
    if "sizes" not in {r[1] for r in c.execute("PRAGMA table_info(plants)")}:
        c.execute("ALTER TABLE plants ADD COLUMN sizes TEXT NOT NULL DEFAULT '[]'")
    if "old_sims" not in {r[1] for r in c.execute("PRAGMA table_info(candidates)")}:
        c.execute("ALTER TABLE candidates ADD COLUMN old_sims TEXT NOT NULL DEFAULT '[]'")
    return c


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def j(v) -> str:
    return json.dumps(v)
