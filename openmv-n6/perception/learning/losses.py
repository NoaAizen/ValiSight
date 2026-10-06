"""Hungarian-matched detection loss with explicit supervision masks."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from perception.student_data import LABEL_NEGATIVE, LABEL_POSITIVE

def detection_loss(output, batch, lambda_center=3.0, lambda_box=1.0,
                   lambda_object=1.0, negative_weight=1.0,
                   hard_negative_weight=1.0):
    pred_obj, pred_boxes = output["object_logits"], output["boxes"]
    presence = batch["gt_presence"]
    gt_boxes = batch["gt_boxes"]
    gt_conf = batch["gt_confidence"]
    states = batch["supervision_state"]
    bsz, queries = pred_obj.shape

    object_targets = torch.zeros_like(pred_obj)
    object_weights = (states >= 0).float().view(-1, 1).expand_as(pred_obj).clone()
    # Verified-empty frames are rare (v3: 17 positive frames per negative), and
    # an unweighted loss is minimised by calling "person" always - measured on
    # v3's first radar run: recall .99 but a 88% false-positive rate on the
    # held-out empty session. Scale the only frames that carry "there is nobody
    # here" so the two classes contribute comparably.
    if negative_weight != 1.0:
        neg = (states == LABEL_NEGATIVE).float().view(-1, 1)
        object_weights = object_weights * (1.0 + (negative_weight - 1.0) * neg)
    # A hard-negative frame can still contain a real person. The Hungarian
    # matches below overwrite the weight of positive queries, so this only
    # raises the cost of the remaining no-object queries (for example V6's
    # smoke/lamp box) without teaching that the real person is background.
    if hard_negative_weight != 1.0 and "thermal_hard_negative" in batch:
        hard = batch["thermal_hard_negative"].float().view(-1, 1)
        object_weights = object_weights * (
            1.0 + (hard_negative_weight - 1.0) * hard)
    total_center = pred_obj.new_tensor(0.0)
    total_box = pred_obj.new_tensor(0.0)
    box_weight_sum = pred_obj.new_tensor(0.0)

    for b in range(bsz):
        if int(states[b]) != LABEL_POSITIVE:
            continue
        gt_idx = torch.where(presence[b] > 0.5)[0]
        if not len(gt_idx):
            continue
        gt = gt_boxes[b, gt_idx]
        cost = torch.cdist(pred_boxes[b], gt, p=1)
        cost -= 0.25 * torch.sigmoid(pred_obj[b])[:, None]
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        p_idx = torch.as_tensor(rows, device=pred_obj.device, dtype=torch.long)
        g_idx = gt_idx[torch.as_tensor(
            cols, device=pred_obj.device, dtype=torch.long)]
        conf = gt_conf[b, g_idx].clamp(0.25, 1.0)
        object_targets[b, p_idx] = 1.0
        object_weights[b, p_idx] = 4.0 * conf
        pbox, gbox = pred_boxes[b, p_idx], gt_boxes[b, g_idx]
        total_center += (torch.abs(pbox[:, :2] - gbox[:, :2]).mean(1)
                         * conf).sum()
        total_box += (F.smooth_l1_loss(
            pbox, gbox, reduction="none").mean(1) * conf).sum()
        box_weight_sum += conf.sum()

    object_raw = F.binary_cross_entropy_with_logits(
        pred_obj, object_targets, reduction="none")
    object_loss = (object_raw * object_weights).sum() \
        / object_weights.sum().clamp(min=1.0)
    center_loss = total_center / box_weight_sum.clamp(min=1.0)
    box_loss = total_box / box_weight_sum.clamp(min=1.0)
    total = (lambda_object * object_loss
             + lambda_center * center_loss + lambda_box * box_loss)
    return {"total": total, "object": object_loss,
            "center": center_loss, "box": box_loss}


