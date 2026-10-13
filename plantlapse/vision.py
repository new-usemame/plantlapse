"""Models: OWLv2 finds plants (open vocabulary), DINOv2 describes what each one looks like.

Both are Apache-2.0 and run on CPU. They are loaded inside the short-lived tick process and
released when it exits, so between captures the service holds no model memory at all.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Discovery asks "is this a plant someone put here?"; the negatives keep lawns, hedges and the
# garden's own shrubs from reading as one. Measured on a porch camera: potted plants scored
# 1.4-5x their best negative, in-ground shrubs 0.4-1.3x.
POSITIVE = ["a potted plant", "a plant in a pot", "a flowering plant", "a houseplant"]
NEGATIVE = ["a bush", "a shrub", "a hedge", "a tree", "a lawn"]
QUERIES = POSITIVE + NEGATIVE
DETECTOR = "google/owlv2-base-patch16-ensemble"
EMBEDDER = "facebook/dinov2-small"

Box = tuple[float, float, float, float]


def sun_elevation(when, lat: float, lon: float) -> float:
    """Solar elevation in degrees (NOAA approximation, good to well under a degree)."""
    import datetime as dt
    import math
    u = when.astimezone(dt.timezone.utc)
    doy = u.timetuple().tm_yday
    hours = u.hour + u.minute / 60 + u.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((hours * 60 + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    cosz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cosz))))


def lighting_mode(img: Image.Image, when=None, latlon: tuple[float, float] | None = None) -> str:
    """'day' in natural daylight, else 'night'.

    With a location, decided by the sun: a camera under a porch light keeps shooting in colour
    all night, and lamplight changes a plant's look enough to read as a different plant.
    Without one, a frame with no colour (infrared) is night.
    """
    if when is not None and latlon is not None:
        return "day" if sun_elevation(when, *latlon) > 3.0 else "night"
    small = np.asarray(img.convert("HSV").resize((160, 90)), dtype=np.float32)
    return "night" if float(small[..., 1].mean()) < 12.0 else "day"


def sharpness(img: Image.Image) -> float:
    import cv2
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _inside(small: Box, big: Box) -> float:
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = (small[2] - small[0]) * (small[3] - small[1])
    return ix * iy / a if a > 0 else 0.0


class Models:
    def __init__(self, threads: int = 4) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, Owlv2ForObjectDetection, Owlv2Processor

        torch.set_num_threads(threads)
        self.torch = torch
        self.op = Owlv2Processor.from_pretrained(DETECTOR)
        self.om = Owlv2ForObjectDetection.from_pretrained(DETECTOR).eval()
        self.ep = AutoImageProcessor.from_pretrained(EMBEDDER)
        self.em = AutoModel.from_pretrained(EMBEDDER).eval()

    def _run(self, img: Image.Image, regions: list[tuple[int, int, int, int]]) -> list[tuple[Box, float, float]]:
        """All regions in one batch. Returns (box in frame coords, positive score, negative score)."""
        torch = self.torch
        crops = [img.crop(r) for r in regions]
        sides = [max(c.size) for c in crops]
        inp = self.op(text=[QUERIES] * len(crops), images=crops, return_tensors="pt")
        with torch.no_grad():
            out = self.om(**inp)
        probs = out.logits.sigmoid()
        boxes = self.op.image_processor.post_process_object_detection(
            out, threshold=0.0, target_sizes=torch.tensor([[s, s] for s in sides]))
        res = []
        npos = len(POSITIVE)
        for i, (reg, crop) in enumerate(zip(regions, crops)):
            pos = probs[i, :, :npos].max(-1).values
            neg = probs[i, :, npos:].max(-1).values
            keep = (pos >= 0.05).nonzero().flatten().tolist()
            bx = boxes[i]["boxes"]
            for k in keep:
                x0, y0, x1, y1 = bx[k].tolist()
                x0, x1 = max(0.0, x0), min(float(crop.size[0]), x1)
                y0, y1 = max(0.0, y0), min(float(crop.size[1]), y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                res.append(((x0 + reg[0], y0 + reg[1], x1 + reg[0], y1 + reg[1]), float(pos[k]), float(neg[k])))
        return res

    def detect(self, img: Image.Image, focus: list[Box] | None = None, *, plant_min: float = 0.20,
               plant_ratio: float = 1.25, proposal_min: float = 0.05,
               proposals_per_focus: int = 12) -> list[tuple[Box, float, str]]:
        """Returns (box, score, kind). kind 'plant' may found a new identity; 'proposal' may only
        be matched to a plant already known (by appearance), so a half-hidden plant stays tracked.

        OWLv2 sees a 960 px square, so a small plant in a 1080p frame is half its size in a
        whole-frame pass. Square tiles cover the frame at native scale, and a close-up around
        each known plant's last box magnifies it further.
        """
        W, H = img.size
        tiles: list[tuple[int, int, int, int]] = []
        if W > H:
            n = max(2, round(W / H) + 1)
            for i in range(n):
                x0 = round(i * (W - H) / (n - 1))
                tiles.append((x0, 0, x0 + H, H))
        else:
            tiles.append((0, 0, W, H))
        focus_regs: list[tuple[int, int, int, int]] = []
        for b in focus or []:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            side = int(min(max(360, 2.6 * max(b[2] - b[0], b[3] - b[1])), min(W, H)))
            x0 = int(min(max(0, cx - side / 2), W - side))
            y0 = int(min(max(0, cy - side / 2), H - side))
            focus_regs.append((x0, y0, x0 + side, y0 + side))
        raw = self._run(img, tiles + focus_regs)
        frame_area = W * H

        def sane(b: Box) -> bool:
            w, h = b[2] - b[0], b[3] - b[1]
            return w * h <= 0.15 * frame_area and min(w, h) >= 32 and 0.2 <= w / h <= 5

        plants = [(b, p) for b, p, n in raw if sane(b) and p >= plant_min and p >= plant_ratio * n]
        kept: list[tuple[Box, float, str]] = []
        for b, sc in sorted(plants, key=lambda t: -t[1]):
            if any(_iou(b, k) > 0.45 for k, _, _ in kept):
                continue
            if any(_inside(b, k) > 0.8 and ks >= 0.6 * sc for k, ks, _ in kept):
                continue
            kept.append((b, sc, "plant"))
        # Proposals: the best few boxes near each known plant, whatever the class score says.
        for fb in focus or []:
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            reach = 2.0 * max(fb[2] - fb[0], fb[3] - fb[1])
            near = [(b, p) for b, p, n in raw if sane(b) and p >= proposal_min
                    and abs((b[0] + b[2]) / 2 - fcx) <= reach and abs((b[1] + b[3]) / 2 - fcy) <= reach]
            added = 0
            for b, sc in sorted(near, key=lambda t: -t[1]):
                if any(_iou(b, k) > 0.6 for k, _, _ in kept):
                    continue
                kept.append((b, sc, "proposal"))
                added += 1
                if added >= proposals_per_focus:
                    break
        return kept

    def embed(self, img: Image.Image, boxes: list[Box]) -> list[np.ndarray]:
        if not boxes:
            return []
        torch = self.torch
        crops = [img.crop(tuple(int(round(v)) for v in b)) for b in boxes]
        with torch.no_grad():
            o = self.em(**self.ep(images=crops, return_tensors="pt"))
        h = o.last_hidden_state
        v = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return [x.astype(np.float32) for x in v]
