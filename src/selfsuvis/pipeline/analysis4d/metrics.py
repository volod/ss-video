"""Deterministic scores for the pinned 4D fixture corpus.

Tracking scores are single-threshold HOTA@0.5 and an IDF1 from majority-vote
identity assignment on the fixture timestamps. They are not TrackEval's
multi-threshold implementation. Depth error on a relative-scale mission is
computed after dividing each series by its median.
"""

from selfsuvis.pipeline.analysis4d.geometry import (
    box_iou_2d,
    box_iou_3d,
    center_error_m,
    mask_boundary_f,
    mask_iou,
)
from selfsuvis.pipeline.analysis4d.schemas import MetricReport
from selfsuvis.pipeline.analysis4d.validate import Bundle

_IOU = 0.5
_ACTIVE = frozenset({"tentative", "confirmed"})
_ACCEPTED = frozenset({"accepted", "corrected"})


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return inter / union


def _coverage(bundle: Bundle) -> float | None:
    assert bundle.truth is not None
    boundaries = bundle.truth.keyframe_boundaries_sec
    if not boundaries:
        return None
    keyframes = bundle.manifest.keyframes_sec
    covered = 0
    for boundary in boundaries:
        if any(abs(frame - boundary) <= 0.5 for frame in keyframes):
            covered += 1
    return covered / len(boundaries)


def _detections(bundle: Bundle) -> tuple[list[dict], list[dict]]:
    assert bundle.truth is not None
    pred = [
        {
            "t": row.t_sec,
            "id": row.track_id,
            "box": row.box.xywh_norm,
            "confidence": row.confidence,
            "label": row.label_normalized,
        }
        for row in bundle.tracks
        if row.state in _ACTIVE
    ]
    truth = [
        {"t": row.t_sec, "id": row.gt_id, "box": row.xywh_norm, "label": row.label}
        for row in bundle.truth.tracks
    ]
    return pred, truth


def _match_frames(pred: list[dict], truth: list[dict]) -> tuple[int, int, int, list[tuple]]:
    times = sorted({row["t"] for row in pred} | {row["t"] for row in truth})
    tp = fp = fn = 0
    matches: list[tuple] = []
    for stamp in times:
        preds = [row for row in pred if row["t"] == stamp]
        gts = [row for row in truth if row["t"] == stamp]
        pairs = []
        for i, left in enumerate(preds):
            for j, right in enumerate(gts):
                pairs.append((box_iou_2d(left["box"], right["box"]), i, j))
        pairs.sort(reverse=True)
        used_p: set[int] = set()
        used_g: set[int] = set()
        for iou, i, j in pairs:
            if iou < _IOU or i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            matches.append((stamp, preds[i]["id"], gts[j]["id"], iou))
        tp += len(used_p)
        fp += len(preds) - len(used_p)
        fn += len(gts) - len(used_g)
    return tp, fp, fn, matches


def _grounding(
    pred: list[dict], truth: list[dict], tp: int, fn: int
) -> tuple[float | None, float | None]:
    gt_count = len(truth)
    if gt_count == 0 and not pred:
        return None, None
    recall = tp / gt_count if gt_count else 0.0
    if not pred:
        return 0.0, recall
    ordered = sorted(pred, key=lambda row: row["confidence"], reverse=True)
    matched_truth: set[tuple] = set()
    hits = 0
    precision_steps = []
    for index, row in enumerate(ordered, start=1):
        candidates = [
            item
            for item in truth
            if item["t"] == row["t"] and (item["t"], item["id"]) not in matched_truth
        ]
        best = None
        best_iou = 0.0
        for item in candidates:
            iou = box_iou_2d(row["box"], item["box"])
            if iou > best_iou:
                best_iou = iou
                best = item
        if best is not None and best_iou >= _IOU:
            matched_truth.add((best["t"], best["id"]))
            hits += 1
        precision_steps.append((hits / index, hits / gt_count if gt_count else 0.0))
    average = 0.0
    previous_recall = 0.0
    for precision, point_recall in precision_steps:
        average += precision * max(0.0, point_recall - previous_recall)
        previous_recall = point_recall
    return average, recall


def _association(matches: list[tuple]) -> tuple[float, int, int]:
    """Return association accuracy, identity switches, and disagreed matches."""
    by_pred: dict[str, list[tuple[float, str]]] = {}
    for stamp, pred_id, gt_id, _iou in matches:
        by_pred.setdefault(pred_id, []).append((stamp, gt_id))
    assignment: dict[str, str] = {}
    switches = 0
    for pred_id, rows in by_pred.items():
        rows.sort()
        counts: dict[str, int] = {}
        previous = None
        for _stamp, gt_id in rows:
            counts[gt_id] = counts.get(gt_id, 0) + 1
            if previous is not None and gt_id != previous:
                switches += 1
            previous = gt_id
        assignment[pred_id] = max(counts, key=lambda key: counts[key])
    agreed = sum(1 for _stamp, pred_id, gt_id, _iou in matches if assignment[pred_id] == gt_id)
    return agreed / len(matches), switches, len(matches) - agreed


def _hota_idf1(
    tp: int, fp: int, fn: int, matches: list[tuple]
) -> tuple[float | None, float | None, float | None]:
    if tp + fp + fn == 0:
        return None, None, None
    det_a = tp / (tp + fp + fn)
    if not matches:
        return 0.0, 0.0, 0.0
    ass_a, switches, disagree = _association(matches)
    idtp = len(matches) - disagree
    idfp = fp + disagree
    idfn = fn + disagree
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn)
    return (det_a * ass_a) ** 0.5, idf1, float(switches)


def _counts(bundle: Bundle) -> float | None:
    assert bundle.truth is not None
    if not bundle.truth.counts:
        return None
    errors = []
    for row in bundle.truth.counts:
        predicted = sum(
            1
            for track in bundle.tracks
            if track.t_sec == row.t_sec
            and track.label_normalized == row.label
            and track.state in _ACTIVE
        )
        errors.append(abs(predicted - row.count))
    return _mean([float(value) for value in errors])


def _masks(bundle: Bundle) -> tuple[float | None, float | None]:
    assert bundle.truth is not None
    if not bundle.truth.masks:
        return None, None
    by_key = {(mask.track_id, mask.t_sec): mask for mask in bundle.masks}
    j_scores = []
    f_scores = []
    for truth in bundle.truth.masks:
        pred = by_key.get((truth.track_id, truth.t_sec))
        if pred is None:
            j_scores.append(0.0)
            f_scores.append(0.0)
            continue
        j_scores.append(mask_iou(pred.bits, truth.bits))
        f_scores.append(mask_boundary_f(pred.bits, pred.width, pred.height, truth.bits))
    return _mean(j_scores), _mean(f_scores)


def _scale_align(values: list[float], relative: bool) -> list[float]:
    if not relative or not values:
        return values
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    if median == 0:
        return values
    return [value / median for value in values]


def _depth(bundle: Bundle) -> float | None:
    assert bundle.truth is not None
    if not bundle.truth.depths or bundle.manifest.coordinate_frame.metric_scale == "unavailable":
        return None
    relative = bundle.manifest.coordinate_frame.metric_scale == "relative"
    pred = {
        (row.subject_id, row.t_sec): row.depth_m
        for row in bundle.geometry
        if row.depth_m is not None
    }
    pairs = []
    for row in bundle.truth.depths:
        if (row.subject_id, row.t_sec) in pred:
            pairs.append((pred[(row.subject_id, row.t_sec)], row.depth_m))
    if not pairs:
        return None
    left = _scale_align([item[0] for item in pairs], relative)
    right = _scale_align([item[1] for item in pairs], relative)
    return _mean([abs(a - b) for a, b in zip(left, right, strict=True)])


def _boxes(bundle: Bundle) -> tuple[float | None, float | None]:
    assert bundle.truth is not None
    if not bundle.truth.boxes or bundle.manifest.coordinate_frame.metric_scale != "metric":
        return None, None
    pred = {(row.subject_id, row.t_sec): row for row in bundle.geometry}
    ious = []
    centers = []
    for row in bundle.truth.boxes:
        sample = pred.get((row.subject_id, row.t_sec))
        if sample is None:
            ious.append(0.0)
            continue
        ious.append(box_iou_3d(sample.center_m, sample.extent_m, row.center_m, row.extent_m))
        centers.append(center_error_m(sample.center_m, row.center_m))
    return _mean(ious), _mean(centers)


def _relations(bundle: Bundle) -> tuple[float | None, float | None]:
    assert bundle.truth is not None
    edges = [
        delta.edge
        for delta in bundle.deltas
        if delta.edge is not None and delta.edge.verification_status in _ACCEPTED
    ]
    truth = bundle.truth.relations
    if not edges and not truth:
        return None, None
    used: set[int] = set()
    matched = 0
    ious = []
    for edge in edges:
        best = None
        best_iou = 0.0
        for index, row in enumerate(truth):
            if index in used:
                continue
            if (edge.subject_id, edge.predicate, edge.object_id) != (
                row.subject_id,
                row.predicate,
                row.object_id,
            ):
                continue
            iou = _temporal_iou(edge.start_sec, edge.end_sec, row.start_sec, row.end_sec)
            if iou > best_iou:
                best_iou = iou
                best = index
        if best is not None and best_iou > 0:
            used.add(best)
            matched += 1
            ious.append(best_iou)
    precision = matched / len(edges) if edges else 0.0
    recall = matched / len(truth) if truth else 0.0
    return precision, recall


def _events(bundle: Bundle) -> tuple[float | None, float | None]:
    assert bundle.truth is not None
    pred = [event for event in bundle.timeline.events if event.verification.status in _ACCEPTED]
    truth = bundle.truth.events
    if not pred and not truth:
        return None, None
    used: set[int] = set()
    matched = 0
    ious = []
    for event in pred:
        participants = set(event.participants)
        best = None
        best_iou = 0.0
        for index, row in enumerate(truth):
            if index in used or event.type != row.type or participants != set(row.participants):
                continue
            iou = _temporal_iou(event.start_sec, event.end_sec, row.start_sec, row.end_sec)
            if iou > best_iou:
                best_iou = iou
                best = index
        if best is not None and best_iou > 0:
            used.add(best)
            matched += 1
            ious.append(best_iou)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(truth) if truth else 0.0
    return _f1(precision, recall), _mean(ious)


def _verifier(bundle: Bundle) -> tuple[float | None, float | None]:
    assert bundle.truth is not None
    if not bundle.truth.verifier:
        return None, None
    by_id = {row.proposal_id: row for row in bundle.proposals}
    negatives = [row for row in bundle.truth.verifier if not row.should_accept]
    positives = [row for row in bundle.truth.verifier if row.should_accept]
    false_accept = 0
    for row in negatives:
        proposal = by_id.get(row.proposal_id)
        if proposal is not None and proposal.verification_status in _ACCEPTED:
            false_accept += 1
    false_reject = 0
    for row in positives:
        proposal = by_id.get(row.proposal_id)
        if proposal is not None and proposal.verification_status == "rejected":
            false_reject += 1
    fa = false_accept / len(negatives) if negatives else None
    fr = false_reject / len(positives) if positives else None
    return fa, fr


def _qa(bundle: Bundle) -> float | None:
    assert bundle.truth is not None
    expected = {row.qa_id: row.answer_value for row in bundle.truth.qa}
    accepted = [row for row in bundle.timeline.qa_pairs if row.verification_status in _ACCEPTED]
    if not accepted:
        return None
    correct = sum(1 for row in accepted if expected.get(row.qa_id) == row.answer.value)
    return correct / len(accepted)


def score_case(bundle: Bundle, *, artifact_bytes: int) -> MetricReport:
    """Score one validated mission that has ``truth``.

    Args:
        bundle: Validated mission.
        artifact_bytes: Total bytes of the mission directory.

    Returns:
        Metric fields. Operational run fields stay null at case scope.
    """
    if bundle.truth is None:
        raise ValueError("score_case requires truth.json")
    pred, truth = _detections(bundle)
    tp, fp, fn, matches = _match_frames(pred, truth)
    ap, recall = _grounding(pred, truth, tp, fn)
    hota, idf1, switches = _hota_idf1(tp, fp, fn, matches)
    mask_j, mask_f = _masks(bundle)
    box_iou, box_center = _boxes(bundle)
    relation_p, relation_r = _relations(bundle)
    event_f1, event_iou = _events(bundle)
    false_accept, false_reject = _verifier(bundle)
    queue_depth = max((stage.queue_depth for stage in bundle.manifest.stages), default=0)
    return MetricReport(
        keyframe_event_coverage=_round_metric(_coverage(bundle)),
        grounding_ap=_round_metric(ap),
        grounding_recall=_round_metric(recall),
        count_mae=_round_metric(_counts(bundle)),
        mask_j=_round_metric(mask_j),
        mask_f=_round_metric(mask_f),
        hota=_round_metric(hota),
        idf1=_round_metric(idf1),
        identity_switches=switches,
        depth_error=_round_metric(_depth(bundle)),
        box_iou=_round_metric(box_iou),
        box_center_error=_round_metric(box_center),
        relation_precision=_round_metric(relation_p),
        relation_recall=_round_metric(relation_r),
        event_f1=_round_metric(event_f1),
        event_temporal_iou=_round_metric(event_iou),
        verifier_false_accept_rate=_round_metric(false_accept),
        verifier_false_reject_rate=_round_metric(false_reject),
        qa_accuracy=_round_metric(_qa(bundle)),
        queue_depth=float(queue_depth),
        artifact_size_bytes=float(artifact_bytes),
    )
