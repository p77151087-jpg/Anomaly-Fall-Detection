from ..io_utils import ContractError


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def window_coverage(rows):
    if any(row.get("n_scheduled_endpoints") is None for row in rows):
        return None
    return divide(
        sum(r.get("n_valid_endpoints", 0) for r in rows),
        sum(r["n_scheduled_endpoints"] for r in rows),
    )


def sequence_metrics(rows):
    table = [[0, 0, 0], [0, 0, 0]]
    ids = [r["sequence_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ContractError("Evaluation requires exactly one result per original sequence")
    for row in rows:
        label, prediction = row["label"], row["prediction"]
        if label not in (0, 1) or prediction not in (None, 0, 1):
            raise ContractError("Invalid sequence label/prediction")
        table[label][2 if prediction is None else prediction] += 1
    tn, fp, _normal_unknown = table[0]
    fn, tp, fall_unknown = table[1]
    total, evaluable = len(rows), tp + tn + fp + fn
    metrics = {
        "metric_unit": "sequence",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n_total": total,
        "n_evaluable": evaluable,
        "n_unknown": total - evaluable,
        "sequence_coverage": divide(evaluable, total),
        "sequence_accuracy": divide(tp + tn, evaluable),
        "sequence_fall_precision": (divide(tp, tp + fp) if tp + fp else 0.0) if evaluable else None,
        "sequence_fall_recall": divide(tp, tp + fn),
        "sequence_fall_f1": (divide(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        if evaluable
        else None,
        "operational_accuracy": divide(tp + tn, total),
        "end_to_end_fall_recall": divide(tp, tp + fn + fall_unknown),
        "outcomes_with_unknown": table,
        "outcome_columns": ["normal", "suspected_fall", "unknown"],
        "outcome_rows": ["ADL", "Fall"],
        "by_class": {},
    }
    for label, name in [(0, "ADL"), (1, "Fall")]:
        group = [r for r in rows if r["label"] == label]
        valid = sum(r["prediction"] is not None for r in group)
        metrics["by_class"][name] = {
            "n_total": len(group),
            "n_unknown": len(group) - valid,
            "sequence_coverage": divide(valid, len(group)),
            "window_coverage": window_coverage(group),
            "n_missing_timelines": sum(r.get("n_scheduled_endpoints") is None for r in group),
        }
    metrics["window_coverage"] = window_coverage(rows)
    metrics["n_missing_timelines"] = sum(r.get("n_scheduled_endpoints") is None for r in rows)
    return metrics
