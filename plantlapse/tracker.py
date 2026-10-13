"""Identity and framing. Pure logic: no models, no disk, so it can be tested on its own.

Identity is decided by APPEARANCE. Each plant keeps a small memory of how it has looked
(embedding prototypes from daylight frames). Every capture, detections are assigned to plants
by a global optimum (Hungarian) over a cost that is mostly appearance distance plus a small
position term. The position term only breaks ties between look-alikes; it is too small to
stop a plant from being recognised after it was carried across the frame, or two plants from
being recognised after they swap places.

Night (infrared) frames carry no colour, and the daylight memory does not transfer to them,
so at night a plant is only re-found near where it already is, and no identity is ever
created or changed. A plant moved at night is re-identified at the next daylight capture.

Framing keeps every plant's timelapse steady: a fixed square crop, centred on an anchor that
ignores detector jitter and wind sway (a dead band), and moves only when the plant has really
moved (one clearly confident sighting far away, or two consecutive ones outside the band).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]


@dataclass
class Params:
    w_pos: float = 0.15          # weight of position in the assignment cost (appearance weight is 1)
    pos_scale: float = 0.25      # distances are measured in units of this fraction of the frame diagonal
    sim_gate: float = 0.50       # below this appearance similarity a detection can never be this plant
    proposal_gate: float = 0.60  # stricter gate for close-up proposals the class score did not vouch for
    sim_confident: float = 0.62  # at or above this, a far-away sighting re-anchors immediately
    night_gate: float = 0.6      # at night, a detection must be within this many crop-sides of the anchor
    fill: float = 1.55           # crop side = fill × the plant's longer box side
    min_side: float = 160.0      # never crop tighter than this many source pixels
    deadband: float = 0.14       # centre may wander this fraction of the crop side before it counts
    move_now: float = 0.45       # a confident sighting this far away (× side) re-anchors at once
    grow: float = 1.18           # the median desired side must pass side × grow to enlarge the crop,
    shrink: float = 0.72         # or fall below side × shrink to shrink it
    size_window: int = 8         # desired sides remembered (2 h at 15 min): a plant does not change size
                                 # within that, so one or two odd boxes cannot zoom the timelapse
    max_protos: int = 48
    proto_novelty: float = 0.92  # only remember a view less similar than this to every stored one
    promote_hits: int = 3        # daylight sightings before a new thing becomes a plant
    candidate_sim: float = 0.60  # a detection continues a candidate at or above this similarity
    candidate_ttl: int = 12      # daylight captures a candidate may go unseen before it is dropped
    duplicate_sim: float = 0.70  # an unmatched detection this similar to a plant is that plant again
    better_box: float = 1.5      # a free box scoring this many times a matched plant's box it overlaps is
                                 # a better box (a real plant), not a second view of that plant
    better_floor: float = 0.1
    part_area: float = 0.35      # a box at most this fraction of a matched plant's area, centred on or
    part_reach: float = 0.25     # within this fraction of its size from that box, is part of the plant
    scenery_sim: float = 0.70    # mean likeness to the spot a day earlier at or above which a candidate is scenery
    w_size: float = 0.3          # cost per unit |ln(size ratio)| between a detection and the plant's last box
    max_size_ratio: float = 2.0  # a box this much bigger or smaller than the plant's last box is not it


@dataclass
class Det:
    box: Box
    score: float
    vec: np.ndarray | None = None
    kind: str = "plant"  # "proposal": may re-find a known plant, never found a new one
    old_sim: float | None = None  # likeness to the same spot a day earlier (None: no frame to compare)


@dataclass
class Plant:
    id: str
    anchor: tuple[float, float] | None = None
    size: float | None = None
    protos: list[np.ndarray] = field(default_factory=list)
    last_box: Box | None = None
    miss_streak: int = 0
    displaced_streak: int = 0
    sizes: list[float] = field(default_factory=list)  # recent desired sides, newest last
    score: float | None = None  # running detector class score from its own (non-proposal) sightings
    new_protos: list[np.ndarray] = field(default_factory=list)


@dataclass
class Candidate:
    id: int | None
    box: Box
    vec: np.ndarray
    hits: int
    first_ts: str
    last_ts: str
    stale: int = 0
    boxes: list[Box] = field(default_factory=list)
    score: float | None = None
    old_sims: list[float] = field(default_factory=list)

    @property
    def scenery(self) -> bool:
        """Its spot looked the same a day before it was first seen, so it was already there: scenery
        (a garden shrub), not a plant someone brought in."""
        return bool(self.old_sims) and sum(self.old_sims) / len(self.old_sims) >= Params().scenery_sim


@dataclass
class Match:
    det: int
    sim: float | None


@dataclass
class StepResult:
    matches: dict[str, Match]
    promoted: list[Candidate]
    candidates: list[Candidate]


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def longer_side(b: Box) -> float:
    return max(b[2] - b[0], b[3] - b[1])


def overlaps(a: Box, b: Box, iou_min: float = 0.25, inside_min: float = 0.5) -> bool:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    if inter <= 0:
        return False
    return inter / (aa + ab - inter) >= iou_min or inter / aa >= inside_min or inter / ab >= inside_min


def part_of(small: Box, big: Box, p: Params) -> bool:
    """A much smaller box centred on or just beside `big`: its pot, a leaf cluster, its base. The plant's
    own box wanders frame to frame (one capture it includes the pot, the next it stops above it), so
    containment alone misses these."""
    a_s = (small[2] - small[0]) * (small[3] - small[1])
    a_b = (big[2] - big[0]) * (big[3] - big[1])
    if a_b <= 0 or a_s > p.part_area * a_b:
        return False
    mx, my = p.part_reach * (big[2] - big[0]), p.part_reach * (big[3] - big[1])
    cx, cy = center(small)
    return big[0] - mx <= cx <= big[2] + mx and big[1] - my <= cy <= big[3] + my


def inside(small: Box, big: Box) -> float:
    """Fraction of `small`'s area that lies within `big`."""
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = (small[2] - small[0]) * (small[3] - small[1])
    return ix * iy / a if a > 0 else 0.0


def similarity(vec: np.ndarray | None, protos: list[np.ndarray]) -> float | None:
    if vec is None or not protos:
        return None
    return float(max(float(np.dot(vec, p)) for p in protos))


def assign(plants: list[Plant], dets: list[Det], mode: str, frame_wh: tuple[int, int],
           p: Params) -> dict[str, Match]:
    if not plants or not dets:
        return {}
    all_sims = [[similarity(d.vec, pl.protos) if mode == "day" else None for d in dets] for pl in plants]
    banned: set[tuple[int, int]] = set()
    # Two plants cannot stand in overlapping boxes. When the best assignment puts them there, the
    # plant that looks less like that spot gives up every box overlapping the winner's, and the
    # assignment is redone (after a swap, the other plant's box is often found nested in this one's).
    for _ in range(len(plants) + 1):
        out = _assign_once(plants, dets, mode, frame_wh, p, all_sims, banned)
        clash = None
        items = list(out.items())
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ia, ma), (ib, mb) = items[a], items[b]
                if overlaps(dets[ma.det].box, dets[mb.det].box):
                    clash = (ia, ma, ib, mb)
                    break
            if clash:
                break
        if clash is None:
            break
        ia, ma, ib, mb = clash
        win, loser = (ma, ib) if (ma.sim or 0.0) >= (mb.sim or 0.0) else (mb, ia)
        li = next(i for i, pl in enumerate(plants) if pl.id == loser)
        for k, d in enumerate(dets):
            if overlaps(d.box, dets[win.det].box):
                banned.add((li, k))
    return out


def _assign_once(plants, dets, mode, frame_wh, p, all_sims, banned) -> dict[str, Match]:
    diag = math.hypot(*frame_wh)
    big = 1e6
    cost = np.full((len(plants), len(dets)), big)
    sims: dict[tuple[int, int], float | None] = {}
    for i, pl in enumerate(plants):
        # Has this plant stayed put? True when a box near its anchor looks like it, and looks more
        # like it than like any other plant (after a swap, the box there is the OTHER plant).
        stayed = False
        if mode == "day" and pl.protos and pl.anchor and pl.size:
            for k, d in enumerate(dets):
                s = all_sims[i][k]
                if (i, k) in banned or s is None or s < p.sim_gate \
                        or math.dist(center(d.box), pl.anchor) > p.move_now * pl.size:
                    continue
                if all(s > (all_sims[j][k] or 0.0) for j in range(len(plants)) if j != i):
                    stayed = True
                    break
        for k, d in enumerate(dets):
            if (i, k) in banned:
                continue
            c = center(d.box)
            dist = math.dist(c, pl.anchor) if pl.anchor else 0.0
            if pl.last_box is not None:
                ratio = longer_side(d.box) / max(1.0, longer_side(pl.last_box))
                if not (1 / p.max_size_ratio <= ratio <= p.max_size_ratio):
                    continue
                size_cost = p.w_size * abs(math.log(ratio))
            else:
                size_cost = 0.0
            if mode == "day" and pl.protos:
                s = all_sims[i][k]
                sims[(i, k)] = s
                if s is None or s < (p.proposal_gate if d.kind == "proposal" else p.sim_gate):
                    continue
                if stayed and pl.size and dist > p.move_now * pl.size:
                    continue  # it is still where it was; a look-alike elsewhere is something else
                cost[i, k] = (1 - s) + p.w_pos * min(1.0, dist / (p.pos_scale * diag)) + size_cost
            else:
                # Night, or a plant with no daylight memory yet: position only, and only close by.
                if pl.anchor is None or pl.size is None:
                    continue
                if dist > p.night_gate * pl.size:
                    continue
                sims[(i, k)] = None
                cost[i, k] = 0.5 + dist / pl.size
    rows, cols = linear_sum_assignment(cost)
    out: dict[str, Match] = {}
    for i, k in zip(rows, cols):
        if cost[i, k] >= big:
            continue
        out[plants[i].id] = Match(det=int(k), sim=sims.get((i, k)))
    return out


def update_framing(pl: Plant, box: Box, sim: float | None, p: Params) -> None:
    c = center(box)
    desired = max(p.min_side, p.fill * longer_side(box))
    if pl.anchor is None or pl.size is None:
        pl.anchor, pl.size, pl.sizes = c, desired, [desired]
        pl.displaced_streak = 0
        return
    d = math.dist(c, pl.anchor)
    if d <= p.deadband * pl.size:
        pl.displaced_streak = 0
    elif d >= p.move_now * pl.size and sim is not None and sim >= p.sim_confident:
        # The plant itself moved, maybe nearer the camera: its old sizes no longer apply.
        pl.anchor, pl.displaced_streak, pl.size, pl.sizes = c, 0, desired, [desired]
    else:
        pl.displaced_streak += 1
        if pl.displaced_streak >= 2:
            pl.anchor, pl.displaced_streak = c, 0
    if not pl.sizes:  # carried over from before sizes were remembered: its size held for an hour
        pl.sizes = [pl.size] * (p.size_window // 2)
    pl.sizes = (pl.sizes + [desired])[-p.size_window:]
    typical = float(np.median(pl.sizes))
    if not pl.size * p.shrink <= typical <= pl.size * p.grow:
        pl.size = typical


def remember(pl: Plant, vec: np.ndarray | None, p: Params) -> None:
    if vec is None:
        return
    s = similarity(vec, pl.protos)
    if s is not None and s >= p.proto_novelty:
        return
    pl.protos.append(vec)
    pl.new_protos.append(vec)
    if len(pl.protos) > p.max_protos:
        # Keep the first views (the plant's identity at introduction) and the most recent ones.
        keep = 8
        pl.protos = pl.protos[:keep] + pl.protos[-(p.max_protos - keep):]


def crop_window(pl: Plant, frame_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square crop of the plant's fixed side at its anchor, shifted (never shrunk) to stay in frame."""
    W, H = frame_wh
    assert pl.anchor is not None and pl.size is not None
    s = int(round(min(pl.size, W, H)))
    x0 = int(round(pl.anchor[0] - s / 2))
    y0 = int(round(pl.anchor[1] - s / 2))
    x0 = min(max(0, x0), W - s)
    y0 = min(max(0, y0), H - s)
    return (x0, y0, x0 + s, y0 + s)


def step(plants: list[Plant], candidates: list[Candidate], dets: list[Det], mode: str,
         frame_wh: tuple[int, int], ts: str, p: Params | None = None) -> StepResult:
    p = p or Params()
    matches = assign(plants, dets, mode, frame_wh, p)
    by_id = {pl.id: pl for pl in plants}
    for pid, m in matches.items():
        pl = by_id[pid]
        d = dets[m.det]
        update_framing(pl, d.box, m.sim, p)
        pl.last_box = d.box
        pl.miss_streak = 0
        if d.kind == "plant":
            pl.score = d.score if pl.score is None else 0.8 * pl.score + 0.2 * d.score
        if mode == "day":
            remember(pl, d.vec, p)
    for pl in plants:
        if pl.id not in matches and mode == "day":
            pl.miss_streak += 1

    promoted: list[Candidate] = []
    if mode != "day":
        return StepResult(matches, promoted, candidates)

    used = {m.det for m in matches.values()}
    free = [k for k in range(len(dets))
            if k not in used and dets[k].vec is not None and dets[k].kind == "plant"]
    # Nor may a box sitting on a plant that was just matched (a leaf cluster, the pot alone).
    # A box on a matched plant is a second view of it (the pot, a leaf cluster) unless it scores far
    # higher than that plant usually does: then it is a different, better-seen plant the first one was
    # wrongly drawn around. A close-up proposal's score is not a class score, so the plant's running
    # score stands in for it.
    def plant_score(pid: str, d: Det) -> float:
        known = by_id[pid].score
        return max(d.score if d.kind == "plant" else 0.0, known or 0.0, p.better_floor)
    matched = [(dets[m.det], plant_score(pid, dets[m.det])) for pid, m in matches.items()]
    free = [k for k in free if not any(
        (overlaps(dets[k].box, md.box) or part_of(dets[k].box, md.box, p)) and dets[k].score < p.better_box * sc
        for md, sc in matched)]
    # A free detection that looks like a known plant is a second box on that plant, not a new plant.
    free = [k for k in free
            if not any((similarity(dets[k].vec, pl.protos) or 0) >= p.duplicate_sim for pl in plants)]

    # Continue candidates by appearance (one detection per candidate).
    taken: set[int] = set()
    for cand in candidates:
        best, best_s = None, p.candidate_sim
        for k in free:
            if k in taken:
                continue
            s = float(np.dot(dets[k].vec, cand.vec))
            if s >= best_s:
                best, best_s = k, s
        if best is None:
            cand.stale += 1
            continue
        taken.add(best)
        d = dets[best]
        cand.hits += 1
        cand.stale = 0
        cand.last_ts = ts
        cand.box = d.box
        cand.boxes.append(d.box)
        cand.score = d.score if cand.score is None else max(cand.score, d.score)
        if d.old_sim is not None:
            cand.old_sims.append(d.old_sim)
        v = cand.vec * (cand.hits - 1) + d.vec
        cand.vec = v / np.linalg.norm(v)
    for k in free:
        if k in taken:
            continue
        d = dets[k]
        candidates.append(Candidate(None, d.box, d.vec, 1, ts, ts, 0, [d.box], d.score,
                                    [d.old_sim] if d.old_sim is not None else []))

    keep: list[Candidate] = []
    for cand in candidates:
        if cand.hits >= p.promote_hits:
            if any(overlaps(cand.box, q.box) for q in promoted):
                continue  # two boxes on one new plant in the same capture: keep the first
            promoted.append(cand)
        elif cand.stale <= p.candidate_ttl:
            keep.append(cand)
    return StepResult(matches, promoted, keep)


def plant_from_candidate(pid: str, cand: Candidate, p: Params | None = None) -> Plant:
    p = p or Params()
    pl = Plant(id=pid)
    boxes = cand.boxes or [cand.box]
    med = tuple(float(np.median([b[i] for b in boxes])) for i in range(4))
    update_framing(pl, med, None, p)  # type: ignore[arg-type]
    pl.sizes = [max(p.min_side, p.fill * longer_side(b)) for b in boxes][-p.size_window:]
    pl.last_box = cand.box
    pl.protos = [cand.vec]
    pl.new_protos = [cand.vec]
    pl.score = cand.score
    return pl
