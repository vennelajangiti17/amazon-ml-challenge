"""Train the pairwise matching classifier.

Features:
- Fast itertuples-based pair construction
- Positives from train_ground_truth.tsv
- Hard negatives from blocking candidates
- Validation set evaluation using the official Macro F_0.5 metric
- Automated threshold sweep to find the score-maximizing cut
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from . import config
from .preprocess import normalize_name, normalize_address
from .blocking import generate_candidates
from .features import build_feature_matrix
from .evaluate import macro_f_beta


def load_sources(sample_size: int = 0):
    print("Loading training data sources...")
    s1 = pd.read_csv(config.TRAIN_S1, sep="\t")
    s2 = pd.read_csv(config.TRAIN_S2, sep="\t")
    s3 = pd.read_csv(config.TRAIN_S3, sep="\t")
    gt = pd.read_csv(config.TRAIN_GT, sep="\t", keep_default_na=False)

    if sample_size > 0 and sample_size < len(s1):
        print(f"Sampling {sample_size:,} Source 1 entities for fast iteration...")
        s1 = s1.sample(n=sample_size, random_state=config.RANDOM_STATE).reset_index(drop=True)
        s1_ids = set(s1["entity_id"])
        gt = gt[gt["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)

        needed_m_ids = set()
        for m in gt["matched_entity_ids"]:
            if m:
                needed_m_ids.update(str(m).split(","))

        bg_size = max(sample_size * 5, 20000)
        s2_needed = s2[s2["entity_id"].isin(needed_m_ids)]
        s2_bg = s2.sample(min(bg_size, len(s2)), random_state=config.RANDOM_STATE)
        s2 = pd.concat([s2_needed, s2_bg], ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)

        s3_needed = s3[s3["entity_id"].isin(needed_m_ids)]
        s3_bg = s3.sample(min(bg_size, len(s3)), random_state=config.RANDOM_STATE)
        s3 = pd.concat([s3_needed, s3_bg], ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)
        print(f"Subsampled candidate universe: {len(s2):,} S2 rows, {len(s3):,} S3 rows")

    print("Normalizing names and addresses...")
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].fillna("").apply(normalize_name)
        df["_norm_addr"] = df["business_address"].fillna("").apply(normalize_address)

    return s1, s2, s3, gt



def build_pairs(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    candidates: dict,
) -> pd.DataFrame:
    print("Building candidate training pairs...")
    all_other = pd.concat([s2, s3], ignore_index=True)
    lookup = {r.entity_id: r for r in all_other.itertuples(index=False)}
    s1_lookup = {r.entity_id: r for r in s1.itertuples(index=False)}

    rows = []
    for gt_row in gt.itertuples(index=False):
        s1_id = gt_row.source1_entity_id
        matched = set(x for x in str(gt_row.matched_entity_ids).split(",") if x)
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue

        cand_ids = candidates.get(s1_id, set()) | matched
        for cand_id in cand_ids:
            cand_rec = lookup.get(cand_id)
            if cand_rec is None:
                continue
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "name_a": s1_rec.business_name,
                "name_b": cand_rec.business_name,
                "addr_a": s1_rec.business_address,
                "addr_b": cand_rec.business_address,
                "country_a": s1_rec.country,
                "country_b": cand_rec.country,
                "label": int(cand_id in matched),
            })

    return pd.DataFrame(rows)


def main():
    sample_size = int(os.environ.get("SAMPLE_SIZE", 0))
    s1, s2, s3, gt = load_sources(sample_size=sample_size)

    # Split Source 1 entities for validation so we evaluate on unseen entities
    s1_train, s1_val = train_test_split(
        s1, test_size=0.15, random_state=config.RANDOM_STATE
    )
    s1_train_ids = set(s1_train["entity_id"])
    s1_val_ids = set(s1_val["entity_id"])

    gt_train = gt[gt["source1_entity_id"].isin(s1_train_ids)]
    gt_val = gt[gt["source1_entity_id"].isin(s1_val_ids)]

    print(f"Generating blocking candidates for {len(s1_train):,} train entities...")
    candidates_train = generate_candidates(s1_train, s2, s3)

    print(f"Generating blocking candidates for {len(s1_val):,} validation entities...")
    candidates_val = generate_candidates(s1_val, s2, s3)

    pairs_train = build_pairs(s1_train, s2, s3, gt_train, candidates_train)
    pairs_val = build_pairs(s1_val, s2, s3, gt_val, candidates_val)

    print(f"Train pairs: {len(pairs_train):,} ({pairs_train['label'].sum():,} pos)")
    print(f"Validation pairs: {len(pairs_val):,} ({pairs_val['label'].sum():,} pos)")

    print("Computing train feature matrix...")
    X_train = build_feature_matrix(pairs_train)
    y_train = pairs_train["label"]

    print("Computing validation feature matrix...")
    X_val = build_feature_matrix(pairs_val)
    y_val = pairs_val["label"]

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "min_child_samples": 20,
        "verbosity": -1,
        "seed": config.RANDOM_STATE,
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    model.save_model(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    # Feature importance
    feat_imp = pd.Series(model.feature_importance(), index=X_train.columns).sort_values(ascending=False)
    print("\nFeature Importance Top 10:")
    print(feat_imp.head(10))

    # Evaluate Macro F_0.5 across thresholds
    print("\nEvaluating validation set predictions & sweeping thresholds...")
    pairs_val["pred_prob"] = model.predict(X_val)

    best_thresh = config.MATCH_THRESHOLD
    best_f_beta = -1.0

    for thresh in np.arange(0.30, 0.91, 0.05):
        thresh = round(float(thresh), 2)
        matches = pairs_val[pairs_val["pred_prob"] >= thresh]
        grouped = (
            matches.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .to_dict()
        )
        pred_rows = [
            {"source1_entity_id": s1_id, "matched_entity_ids": grouped.get(s1_id, "")}
            for s1_id in s1_val_ids
        ]
        pred_df = pd.DataFrame(pred_rows)
        score = macro_f_beta(pred_df, gt_val, beta=0.5)
        print(f"  Threshold {thresh:.2f} -> Validation Macro F_0.5: {score:.5f}")
        if score > best_f_beta:
            best_f_beta = score
            best_thresh = thresh

    print(f"\n>>> OPTIMAL MATCH_THRESHOLD: {best_thresh:.2f} (Macro F_0.5: {best_f_beta:.5f}) <<<")


if __name__ == "__main__":
    main()

