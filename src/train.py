"""Train the pairwise matching classifier.

Positives come from train_ground_truth.tsv. Negatives come from the
blocking candidates that are NOT in the ground truth (hard negatives --
records that look superficially similar but aren't matches, which is
exactly what the model needs to learn to reject for a precision-heavy
metric like F_0.5).

Run from the `code/business_entity_resolution` directory:
    python -m src.train
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from . import config
from .preprocess import normalize_name
from .blocking import generate_candidates
from .features import build_feature_matrix


def load_sources():
    s1 = pd.read_csv(config.TRAIN_S1, sep="\t")
    s2 = pd.read_csv(config.TRAIN_S2, sep="\t")
    s3 = pd.read_csv(config.TRAIN_S3, sep="\t")
    gt = pd.read_csv(config.TRAIN_GT, sep="\t", keep_default_na=False)
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].apply(normalize_name)
    return s1, s2, s3, gt


def build_pairs(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                 gt: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    lookup = {r["entity_id"]: r for _, r in pd.concat([s2, s3], ignore_index=True).iterrows()}
    s1_lookup = {r["entity_id"]: r for _, r in s1.iterrows()}

    rows = []
    for _, gt_row in gt.iterrows():
        s1_id = gt_row["source1_entity_id"]
        matched = set(x for x in str(gt_row["matched_entity_ids"]).split(",") if x)
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        # union: true matches (so the model always sees its positives) +
        # blocking candidates (so it sees realistic hard negatives)
        cand_ids = candidates.get(s1_id, set()) | matched
        for cand_id in cand_ids:
            cand_rec = lookup.get(cand_id)
            if cand_rec is None:
                continue
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "name_a": s1_rec["business_name"], "name_b": cand_rec["business_name"],
                "addr_a": s1_rec["business_address"], "addr_b": cand_rec["business_address"],
                "country_a": s1_rec["country"], "country_b": cand_rec["country"],
                "label": int(cand_id in matched),
            })
    return pd.DataFrame(rows)


def main():
    s1, s2, s3, gt = load_sources()
    candidates = generate_candidates(s1, s2, s3)
    print(f"Blocking produced candidates for {len(candidates)}/{len(s1)} Source1 entities")

    pairs = build_pairs(s1, s2, s3, gt, candidates)
    print(f"Built {len(pairs)} training pairs ({pairs['label'].sum()} positive)")

    feats = build_feature_matrix(pairs)
    X, y = feats, pairs["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": config.RANDOM_STATE,
    }
    model = lgb.train(
        params, train_set, num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    model.save_model(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print("Feature importance:", dict(zip(feats.columns, model.feature_importance())))


if __name__ == "__main__":
    main()
