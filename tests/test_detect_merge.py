"""core.detect.merge: rule-based warm blobs x learned person boxes.

Pure logic, synthetic boxes with known overlaps through the public
merge_thermal_boxes() entry point.
"""
from collections import namedtuple

from core.detect.merge import merge_thermal_boxes, iou, MERGE_MIN_IOU

B = namedtuple("B", "x y w h score label", defaults=(None,))


def test_agreeing_pair_boosts_confidence_and_takes_ai_geometry():
    rule = B(60, 40, 30, 50, 0.5)
    ai = B(62, 38, 26, 54, 0.6, "person")
    (m,) = merge_thermal_boxes([rule], [ai])
    assert m.sources == ("rule", "ai")
    assert m.label == "person"
    assert (m.x, m.w) == (62, 26)                       # AI outline wins
    assert abs(m.confidence - (1 - 0.5 * 0.4)) < 1e-9   # noisy-OR 0.8
    assert m.confidence > max(0.5, 0.6)


def test_disjoint_boxes_pass_through_unchanged():
    rule = B(10, 10, 20, 30, 0.4)
    ai = B(120, 60, 25, 40, 0.7, "person")
    out = merge_thermal_boxes([rule], [ai])
    assert len(out) == 2
    by_src = {m.sources: m for m in out}
    assert by_src[("rule",)].confidence == 0.4
    assert by_src[("rule",)].label == "warm"
    assert by_src[("ai",)].confidence == 0.7


def test_below_min_iou_is_not_a_pair():
    rule = B(0, 0, 30, 30, 0.5)
    ai = B(25, 25, 30, 30, 0.5, "person")               # sliver overlap
    assert iou(rule, ai) < MERGE_MIN_IOU
    assert len(merge_thermal_boxes([rule], [ai])) == 2


def test_greedy_pairs_best_overlap_first():
    rule = [B(60, 40, 30, 50, 0.5), B(100, 40, 30, 50, 0.5)]
    ai = [B(98, 42, 30, 48, 0.9, "person")]             # overlaps rule[1] most
    out = merge_thermal_boxes(rule, ai)
    paired = [m for m in out if m.sources == ("rule", "ai")]
    assert len(paired) == 1 and paired[0].x == 98
    assert sum(1 for m in out if m.sources == ("rule",)) == 1


def test_empty_ai_list_is_identity_on_rule_boxes():
    rule = [B(60, 40, 30, 50, 0.5)]
    (m,) = merge_thermal_boxes(rule, [])
    assert m.sources == ("rule",) and m.confidence == 0.5


def test_output_sorted_by_confidence():
    out = merge_thermal_boxes([B(0, 0, 10, 10, 0.3)],
                              [B(100, 0, 10, 10, 0.9, "person")])
    assert [m.confidence for m in out] == sorted(
        (m.confidence for m in out), reverse=True)
