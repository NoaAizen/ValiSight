"""Merge two thermal detection sources into one box list (pure).

The rule-based detector (core.detect.thermal) finds anything warmer than the
background; a learned person detector finds person-SHAPED things. They fail
differently — a radiator defeats the first, a low-contrast person defeats
the second — so agreement is strong evidence and either alone is kept at its
own confidence. This mirrors the sensor-level fusion one level down: two
independent looks at the same frame, combined, never a hard veto.

Score for an agreeing pair is the noisy-OR 1-(1-a)(1-b): both saw it, so the
merged detection is at least as confident as the stronger source. ANALYTIC
SCAFFOLD like every constant in this package — swap for a learned combiner
when there is data to fit one.

Pure module: works on any objects carrying .x .y .w .h and a confidence
(``confidence`` or ``score``); returns MergedBox records. The AI adapter
(torch/cv2.dnn) lives OUTSIDE core and feeds this.
"""
from dataclasses import dataclass

MERGE_MIN_IOU = 0.20     # boxes overlapping at least this much are one target


@dataclass
class MergedBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str            # 'person' when the learned detector saw it
    sources: tuple        # ('rule',), ('ai',) or ('rule', 'ai')


def _conf(b):
    for k in ("confidence", "score"):
        v = getattr(b, k, None)
        if v is not None:
            return float(v)
    raise AttributeError("box %r has no confidence/score" % (b,))


def iou(a, b):
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def merge_thermal_boxes(rule_boxes, ai_boxes, min_iou=MERGE_MIN_IOU):
    """Greedy IoU pairing, best overlap first. Returns [MergedBox] sorted by
    confidence.

    A pair keeps the AI box geometry (a trained detector outlines a person
    more tightly than a warm-blob bound, which bleeds into carried objects)
    and the noisy-OR confidence; unpaired boxes pass through unchanged.
    """
    pairs = sorted(((iou(r, a), ri, ai)
                    for ri, r in enumerate(rule_boxes)
                    for ai, a in enumerate(ai_boxes)),
                   reverse=True)
    used_r, used_a, out = set(), set(), []
    for ov, ri, ai in pairs:
        if ov < min_iou or ri in used_r or ai in used_a:
            continue
        used_r.add(ri)
        used_a.add(ai)
        r, a = rule_boxes[ri], ai_boxes[ai]
        cr, ca = _conf(r), _conf(a)
        out.append(MergedBox(x=a.x, y=a.y, w=a.w, h=a.h,
                             confidence=1.0 - (1.0 - cr) * (1.0 - ca),
                             label=getattr(a, "label", "person") or "person",
                             sources=("rule", "ai")))
    for ri, r in enumerate(rule_boxes):
        if ri not in used_r:
            out.append(MergedBox(r.x, r.y, r.w, r.h, _conf(r),
                                 getattr(r, "label", None) or "warm",
                                 ("rule",)))
    for ai, a in enumerate(ai_boxes):
        if ai not in used_a:
            out.append(MergedBox(a.x, a.y, a.w, a.h, _conf(a),
                                 getattr(a, "label", "person") or "person",
                                 ("ai",)))
    out.sort(key=lambda b: b.confidence, reverse=True)
    return out
