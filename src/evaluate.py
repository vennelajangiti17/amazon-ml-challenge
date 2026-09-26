"""Macro-averaged F_beta scoring, matching the challenge's evaluation formula.

Use this on a validation split carved out of train_ground_truth.tsv (never
on data your model trained on) to tune MATCH_THRESHOLD before you spend a
leaderboard submission.
"""
import pandas as pd


def _ids(cell) -> set:
    return set(x for x in str(cell).split(",") if x)


def f_beta_row(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    if precision == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f_beta(pred_df: pd.DataFrame, gt_df: pd.DataFrame, beta: float = 0.5) -> float:
    """pred_df: source1_entity_id, matched_entity_ids.
    gt_df: source1_entity_id, matched_entity_ids (ground truth)."""
    gt_map = {r["source1_entity_id"]: _ids(r["matched_entity_ids"]) for _, r in gt_df.iterrows()}
    scores = []
    for _, r in pred_df.iterrows():
        pred_ids = _ids(r["matched_entity_ids"])
        true_ids = gt_map.get(r["source1_entity_id"], set())
        scores.append(f_beta_row(pred_ids, true_ids, beta))
    return sum(scores) / len(scores) if scores else 0.0
