"""Identity and framing behaviour, driven with synthetic appearance vectors.

Each plant gets a random unit vector; a sighting is that vector plus noise, so "looks like"
is controlled exactly and the tests exercise the tracker's decisions, not a model's.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from plantlapse.tracker import Candidate, Det, Params, Plant, crop_window, step, plant_from_candidate

RNG = np.random.default_rng(7)
WH = (1920, 1080)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def look(base, noise=0.35):
    return unit(base + noise * unit(RNG.normal(size=base.shape)))


A_LOOK, B_LOOK, C_LOOK = (unit(RNG.normal(size=768)) for _ in range(3))


def box_at(cx, cy, w=140, h=170):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def plant(pid, base, cx, cy):
    pl = plant_from_candidate(pid, Candidate(None, box_at(cx, cy), look(base), 3, "t", "t", 0, [box_at(cx, cy)]))
    for _ in range(6):
        pl.protos.append(look(base))
    return pl


def test_noise_level_is_realistic():
    # Same plant ~0.89 similar, different plants ~0: comparable to (but cleaner than) DINOv2 on real crops.
    same = float(look(A_LOOK) @ look(A_LOOK))
    diff = float(look(A_LOOK) @ look(B_LOOK))
    assert 0.75 < same < 0.97 and abs(diff) < 0.2


def test_jitter_and_sway_do_not_move_the_crop():
    a = plant("a", A_LOOK, 1290, 670)
    win0 = crop_window(a, WH)
    for dx, dy in [(6, -4), (-9, 3), (12, 8), (-5, -10), (0, 7)]:
        step([a], [], [Det(box_at(1290 + dx, 670 + dy), 0.4, look(A_LOOK))], "day", WH, "t")
    assert crop_window(a, WH) == win0


def test_swapped_lookalike_plants_keep_their_identities():
    # Two plants as alike as real crops of different plants get (~0.7 similar): appearance must
    # still beat "they're each where the other one was".
    b_like = unit(0.75 * A_LOOK + 0.66 * B_LOOK)
    a, b = plant("a", A_LOOK, 1290, 670), plant("b", b_like, 640, 655)
    for _ in range(4):
        a.protos.append(look(A_LOOK, 0.2)); b.protos.append(look(b_like, 0.2))
    assert 0.6 < float(look(A_LOOK, 0.2) @ look(b_like, 0.2)) < 0.85
    dets = [Det(box_at(1290, 670), 0.4, look(b_like, 0.2)), Det(box_at(640, 655), 0.4, look(A_LOOK, 0.2))]
    r = step([a, b], [], dets, "day", WH, "t")
    assert r.matches["a"].det == 1 and r.matches["b"].det == 0


def test_swapped_plants_keep_their_identities():
    a, b = plant("a", A_LOOK, 1290, 670), plant("b", B_LOOK, 640, 655)
    dets = [Det(box_at(1290, 670), 0.4, look(B_LOOK)), Det(box_at(640, 655), 0.4, look(A_LOOK))]
    r = step([a, b], [], dets, "day", WH, "t")
    assert r.matches["a"].det == 1 and r.matches["b"].det == 0


def test_a_carried_plant_is_followed_and_reframed_at_once():
    a = plant("a", A_LOOK, 1290, 670)
    r = step([a], [], [Det(box_at(300, 400), 0.4, look(A_LOOK, 0.2))], "day", WH, "t")
    assert "a" in r.matches
    x0, y0, x1, y1 = crop_window(a, WH)
    assert x0 < 300 < x1 and y0 < 400 < y1


def test_a_small_ambiguous_shift_needs_two_sightings():
    a = plant("a", A_LOOK, 1290, 670)
    before = a.anchor
    p = Params()
    shift = (p.deadband + 0.1) * a.size  # outside the dead band, short of an obvious move
    step([a], [], [Det(box_at(1290 + shift, 670), 0.4, look(A_LOOK))], "day", WH, "t")
    assert a.anchor == before
    step([a], [], [Det(box_at(1290 + shift, 670), 0.4, look(A_LOOK))], "day", WH, "t")
    assert a.anchor != before


def test_a_new_plant_gets_its_own_identity_after_three_sightings_only():
    a = plant("a", A_LOOK, 1290, 670)
    cands = []
    for i in range(3):
        r = step([a], cands, [Det(box_at(1290, 670), 0.4, look(A_LOOK)),
                              Det(box_at(400, 600), 0.3, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        if i < 2:
            assert not r.promoted
    assert len(r.promoted) == 1 and r.matches["a"].det == 0


def test_two_new_plants_at_once_become_two_identities():
    cands, promoted = [], []
    for i in range(3):
        r = step([], cands, [Det(box_at(400, 600), 0.3, look(B_LOOK)),
                             Det(box_at(1500, 600), 0.3, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        promoted += r.promoted
    assert len(promoted) == 2


def test_a_second_box_on_a_known_plant_never_becomes_a_plant():
    a = plant("a", A_LOOK, 1290, 670)
    cands = []
    for i in range(5):
        # the pot alone, inside the plant's box, looking somewhat different
        r = step([a], cands, [Det(box_at(1290, 670), 0.4, look(A_LOOK)),
                              Det(box_at(1290, 720, 50, 50), 0.4, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        assert not r.promoted


def test_proposals_never_found_identities():
    cands = []
    for i in range(5):
        r = step([], cands, [Det(box_at(400, 600), 0.08, look(C_LOOK), "proposal")], "day", WH, f"t{i}")
        cands = r.candidates
        assert not r.promoted and not cands


def test_a_lookalike_proposal_does_not_steal_a_hidden_plant():
    a = plant("a", A_LOOK, 1290, 670)
    stranger = unit(A_LOOK * 0.3 + C_LOOK)  # ~0.3 similar: some resemblance, not the plant
    r = step([a], [], [Det(box_at(1100, 600), 0.1, look(stranger, 0.1), "proposal")], "day", WH, "t")
    assert "a" not in r.matches and a.miss_streak == 1


def test_night_never_creates_or_changes_identity_and_stays_put():
    a = plant("a", A_LOOK, 1290, 670)
    win0 = crop_window(a, WH)
    r = step([a], [], [Det(box_at(1290, 670), 0.4, None), Det(box_at(300, 300), 0.4, None)], "night", WH, "t")
    assert not r.promoted and not r.candidates and crop_window(a, WH) == win0
    assert a.miss_streak == 0 or "a" in r.matches


def test_crop_stays_inside_the_frame_at_the_edge():
    a = plant("a", A_LOOK, 30, 1060)
    x0, y0, x1, y1 = crop_window(a, WH)
    assert x0 == 0 and y1 == WH[1] and x1 - x0 == y1 - y0


def test_growth_enlarges_the_crop_slowly_and_only_when_sustained():
    a = plant("a", A_LOOK, 1000, 600)
    for _ in range(10):
        step([a], [], [Det(box_at(1000, 600), 0.4, look(A_LOOK))], "day", WH, "t0")
    s0 = a.size
    for _ in range(3):
        step([a], [], [Det(box_at(1000, 600, 200, 260), 0.4, look(A_LOOK))], "day", WH, "t1")
    assert a.size == s0
    sides = [crop_window(a, WH)[2] - crop_window(a, WH)[0]]
    for _ in range(100):
        step([a], [], [Det(box_at(1000, 600, 200, 260), 0.4, look(A_LOOK))], "day", WH, "t2")
        sides.append(crop_window(a, WH)[2] - crop_window(a, WH)[0])
    assert a.size > 1.3 * s0
    assert all(abs(y - x) <= 0.01 * x + 1 for x, y in zip(sides, sides[1:])), "the crop jumped instead of easing"


REAL = json.loads((Path(__file__).parent / "data" / "real_boxes_2026_10_13.json").read_text())


def replay_real(rows):
    """Found a plant on the first three real boxes, then feed it the rest; returns (ts, window) per capture."""
    boxes = [tuple(r[1:]) for r in rows]
    a = plant_from_candidate("a", Candidate(None, boxes[2], look(A_LOOK), 3, "t", "t", 0, boxes[:3]))
    a.protos += [look(A_LOOK) for _ in range(6)]
    out = []
    for r, b in zip(rows[3:], boxes[3:]):
        step([a], [], [Det(b, 0.4, look(A_LOOK))], "day", WH, r[0])
        out.append((r[0], crop_window(a, WH)))
    return out


def contains(win, box):
    return win[0] <= box[0] and win[1] <= box[1] and win[2] >= box[2] and win[3] >= box[3]


@pytest.mark.parametrize("name, whole_ts", [("carnivorous", "20261013T180000Z"), ("basil", "20261013T130000Z")])
def test_partial_boxes_never_zoom_the_real_plants(name, whole_ts):
    # 2026-10-13: late in the day the detector boxed only the carnivorous plant's pitcher tops (20:15,
    # 21:15) or only its pot (22:30), and the basil's box came and went with the railing post beside it.
    # The live crop shrank from 248 to 172 (and the basil's from 229 to 164 and back to 213), cutting
    # the plants off. The crop must ease, never jump, and keep the whole plant in it.
    rows = REAL[name]
    whole = tuple(next(r for r in rows if r[0] == whole_ts)[1:])
    wins = replay_real(rows)
    sides = [w[2] - w[0] for _, w in wins]
    assert all(abs(y - x) <= 0.01 * x + 1 for x, y in zip(sides, sides[1:])), "the crop jumped"
    clipped = [ts for ts, w in wins if ts >= "20261013" and not contains(w, whole)]
    assert not clipped, f"the whole plant fell outside the crop at {clipped}"


def test_night_boxes_never_reframe():
    a = plant("a", A_LOOK, 1290, 670)
    win0, sizes0 = crop_window(a, WH), list(a.sizes)
    for _ in range(20):
        step([a], [], [Det(box_at(1300, 680, 260, 320), 0.4, None)], "night", WH, "t")
    assert crop_window(a, WH) == win0 and a.sizes == sizes0


def test_two_oversized_boxes_in_a_row_do_not_zoom_the_timelapse():
    # The carnivorous plant, 2026-10-12 15:00-17:15Z: at 15:45 and 16:00 the detector's box reached up and
    # into the hedge, and the crop zoomed out for 45 minutes. A plant does not change size in 30 minutes.
    real = [(1221, 580, 1366, 759), (1218, 593, 1359, 758), (1216, 602, 1356, 758), (1219, 550, 1380, 758),
            (1218, 519, 1402, 759), (1221, 598, 1362, 758), (1218, 606, 1336, 757), (1222, 593, 1360, 761),
            (1225, 602, 1355, 758), (1216, 609, 1327, 758)]
    a = plant_from_candidate("a", Candidate(None, real[2], look(A_LOOK), 3, "t", "t", 0, real[:3]))
    a.protos += [look(A_LOOK) for _ in range(6)]
    win0 = crop_window(a, WH)
    for b in real[3:]:
        step([a], [], [Det(b, 0.4, look(A_LOOK))], "day", WH, "t")
        assert crop_window(a, WH) == win0


def test_a_plant_moved_nearer_the_camera_is_reframed_at_its_new_size_at_once():
    a = plant("a", A_LOOK, 1290, 670)
    for _ in range(7):
        step([a], [], [Det(box_at(1290, 670), 0.4, look(A_LOOK))], "day", WH, "t0")
    step([a], [], [Det(box_at(700, 600, 220, 270), 0.4, look(A_LOOK, 0.2))], "day", WH, "t1")
    x0, y0, x1, y1 = crop_window(a, WH)
    assert x1 - x0 >= 1.5 * 270 - 1 and x0 < 590 and x1 > 810


def test_a_replacement_plant_in_a_removed_plants_spot_is_a_new_plant():
    a = plant("a", A_LOOK, 1290, 670)
    cands, promoted = [], []
    for i in range(3):
        r = step([a], cands, [Det(box_at(1290, 670), 0.4, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        promoted += r.promoted
        assert "a" not in r.matches
    assert len(promoted) == 1


def test_two_boxes_on_one_new_plant_in_one_capture_make_one_plant():
    cands, promoted = [], []
    for i in range(3):
        r = step([], cands, [Det(box_at(400, 600), 0.3, look(C_LOOK)),
                             Det(box_at(400, 640, 120, 120), 0.3, look(B_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        promoted += r.promoted
    assert len(promoted) == 1


def test_sun_elevation_matches_known_values():
    import datetime as dt
    from plantlapse.vision import sun_elevation
    # Greenwich, 2026-06-21 12:00 UTC: ~62 deg; same place at midnight: well below the horizon.
    assert abs(sun_elevation(dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc), 51.48, 0.0) - 62.0) < 1.0
    assert sun_elevation(dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.timezone.utc), 51.48, 0.0) < -10
    # Equator at the March equinox, local noon: nearly overhead.
    assert sun_elevation(dt.datetime(2026, 3, 20, 12, 7, tzinfo=dt.timezone.utc), 0.0, 0.0) > 88


def test_a_lookalike_far_away_cannot_steal_a_plant_that_stayed_put():
    # Plant founded partly on background; later a big background patch looks even more like it.
    a = plant("a", A_LOOK, 1290, 670)
    stay = Det(box_at(1290, 670), 0.4, look(A_LOOK, 0.75))
    far = Det(box_at(300, 300), 0.4, look(A_LOOK, 0.1))
    assert float(stay.vec @ A_LOOK) + 0.15 < float(far.vec @ A_LOOK)  # appearance alone would pick far
    r = step([a], [], [stay, far], "day", WH, "t")
    assert r.matches["a"].det == 0
    assert crop_window(a, WH)[0] < 1290 < crop_window(a, WH)[2]


def test_a_much_better_box_on_a_weak_plant_still_founds_its_own_plant():
    a = plant("a", A_LOOK, 1300, 600)   # a weak, background-ish "plant"
    a.last_box = box_at(1300, 600, 200, 400)
    cands, promoted = [], []
    for i in range(3):
        r = step([a], cands, [Det(box_at(1300, 600, 200, 400), 0.17, look(A_LOOK)),
                              Det(box_at(1280, 690, 110, 160), 0.45, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        promoted += r.promoted
    assert len(promoted) == 1


def test_a_box_of_a_very_different_size_is_not_the_plant():
    a = plant("a", A_LOOK, 1000, 600)
    r = step([a], [], [Det(box_at(1000, 600, 500, 700), 0.4, look(A_LOOK))], "day", WH, "t")
    assert "a" not in r.matches


def test_the_pot_inside_a_plant_matched_by_a_weak_proposal_is_not_a_new_plant():
    a = plant("a", A_LOOK, 1280, 680)
    a.last_box = box_at(1280, 680, 115, 160)
    a.score = 0.45  # its usual class score, from earlier sightings
    cands, promoted = [], []
    for i in range(4):
        r = step([a], cands, [Det(box_at(1280, 680, 115, 160), 0.06, look(A_LOOK), "proposal"),
                              Det(box_at(1294, 727, 46, 48), 0.35, look(C_LOOK))], "day", WH, f"t{i}")
        cands = r.candidates
        promoted += r.promoted
        assert "a" in r.matches
    assert not promoted


def test_something_that_was_already_there_yesterday_is_promoted_as_scenery():
    cands, promoted = [], []
    for i in range(3):
        r = step([], cands, [Det(box_at(520, 620), 0.3, look(A_LOOK), old_sim=0.88)], "day", WH, f"t{i}")
        cands, promoted = r.candidates, promoted + r.promoted
    assert len(promoted) == 1 and promoted[0].scenery


def test_a_plant_brought_in_today_is_not_scenery_and_no_history_means_new():
    for old in (0.45, None):
        cands, promoted = [], []
        for i in range(3):
            r = step([], cands, [Det(box_at(520, 620), 0.3, look(A_LOOK), old_sim=old)], "day", WH, f"t{i}")
            cands, promoted = r.candidates, promoted + r.promoted
        assert len(promoted) == 1 and not promoted[0].scenery


def test_the_pot_base_just_below_a_box_that_stops_above_it_is_part_of_the_plant():
    # Real case (eval, 20:30Z): the plant's box ended at y=714, its pot base was boxed at 702-750 (0.24).
    a = plant("a", A_LOOK, 1300, 650)
    a.last_box = (1227, 590, 1374, 714)
    a.score = 0.49
    cands, promoted = [], []
    for i in range(4):
        r = step([a], cands, [Det((1227, 590, 1374, 714), 0.5, look(A_LOOK)),
                              Det((1271, 703, 1317, 751), 0.24, look(C_LOOK))], "day", WH, f"t{i}")
        cands, promoted = r.candidates, promoted + r.promoted
    assert not promoted


def with_sims(bases, targets):
    """A unit vector whose similarity to each base vector is exactly the target (bases may correlate)."""
    B = np.stack(bases).astype(np.float64)
    w = np.linalg.solve(B @ B.T, np.array(targets))
    v = w @ B
    n = RNG.normal(size=B.shape[1])
    n -= B.T @ np.linalg.lstsq(B.T, n, rcond=None)[0]   # orthogonal to every base
    v = v + np.sqrt(1 - v @ v) * n / np.linalg.norm(n)
    return v.astype(np.float32)


def test_swap_where_the_old_spot_holds_a_nested_box_of_the_other_plant():
    # Real case, sims measured on real pixels (t9 swap, 2026-10-13): at C's old spot stand A's box
    # and a sub-box of it that looks almost as much like C; C itself is now at A's old spot.
    shared = unit(RNG.normal(size=768))
    a_base = unit(shared + unit(RNG.normal(size=768)))
    c_base = unit(shared + unit(RNG.normal(size=768)))   # two porch plants: similarity ~0.5
    a = plant_from_candidate("a", Candidate(None, (1220, 598, 1333, 758), a_base, 3, "t", "t", 0))
    c = plant_from_candidate("c", Candidate(None, (585, 616, 687, 732), c_base, 3, "t", "t", 0))
    a.protos, c.protos = [a_base] * 7, [c_base] * 7
    dets = [Det((566, 570, 707, 730), 0.54, with_sims([a_base, c_base], [0.89, 0.65])),
            Det((600, 570, 707, 693), 0.19, with_sims([a_base, c_base], [0.75, 0.71]), "proposal"),
            Det((1246, 637, 1345, 757), 0.22, with_sims([a_base, c_base], [0.56, 0.79]), "proposal")]
    r = step([a, c], [], dets, "day", WH, "t")
    assert r.matches["a"].det == 0 and r.matches["c"].det == 2


WEEK = json.loads((Path(__file__).parent / "data" / "real_boxes_2026_10_12_to_15.json").read_text())


def test_two_days_of_partial_boxes_do_not_shrink_a_plant_that_has_not_changed():
    # 2026-10-14 to 10-15: the detector boxed only the carnivorous plant's pitcher tops, never the pot,
    # for two straight days. Every recent box asked for the 160 px floor, the 2-day percentile followed
    # them, and the live crop eased from 248 to 193 on its way to cutting the pot off. A plant does not
    # shrink in two days: the crop holds until most of a week says it is smaller.
    rows = WEEK["carnivorous"]
    whole = tuple(next(r for r in rows if r[0] == "20261013T160000Z")[1:])
    wins = replay_real(rows)
    settled = next(w for ts, w in wins if ts >= "20261014T120000Z")
    side = lambda w: w[2] - w[0]
    assert side(wins[-1][1]) >= 0.92 * side(settled), "two days of partial boxes shrank the crop"
    clipped = [ts for ts, w in wins if ts >= "20261013" and not contains(w, whole)]
    assert not clipped, f"the whole plant fell outside the crop at {clipped}"


def test_a_plant_that_really_got_smaller_is_reframed_within_a_week():
    # The other side of the rule above: a pruned plant must not stay loosely framed forever.
    a = plant("a", A_LOOK, 1290, 670)
    for _ in range(100):
        step([a], [], [Det(box_at(1290, 670, 160, 200), 0.4, look(A_LOOK))], "day", WH, "t0")
    big = a.size
    for _ in range(7 * 48):
        step([a], [], [Det(box_at(1290, 670, 100, 120), 0.4, look(A_LOOK))], "day", WH, "t1")
    assert a.size <= 1.08 * 1.55 * 120 < 0.7 * big
