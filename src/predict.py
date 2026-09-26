"""Run inference on the test set and write the two required output files.

Outputs:
1. output/candidate_pairs.tsv — the candidate-generation / blocking set fed to the model
2. output/matching_results.tsv — the final model-scored, thresholded predictions
"""
import os
import pandas as pd
import lightgbm as lgb

from . import config
from .preprocess import normalize_name, normalize_address
from .blocking import generate_candidates
from .features import build_feature_matrix


def load_test_sources():
    print("Loading test data sources...")
    s1 = pd.read_csv(config.TEST_S1, sep="\t")
    s2 = pd.read_csv(config.TEST_S2, sep="\t")
    s3 = pd.read_csv(config.TEST_S3, sep="\t")
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].fillna("").apply(normalize_name)
        df["_norm_addr"] = df["business_address"].fillna("").apply(normalize_address)
    return s1, s2, s3


def main(threshold: float = config.MATCH_THRESHOLD):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    s1, s2, s3 = load_test_sources()

    print(f"Generating candidates for {len(s1):,} test entities...")
    candidates = generate_candidates(s1, s2, s3)

    all_other = pd.concat([s2, s3], ignore_index=True)
    lookup = {r.entity_id: r for r in all_other.itertuples(index=False)}
    s1_lookup = {r.entity_id: r for r in s1.itertuples(index=False)}
    all_s1_ids = list(s1["entity_id"])

    # --- 1. candidate_pairs.tsv (blocking output, one row per Source 1 entity) ---
    print("Preparing candidate pairs output...")
    cand_rows, pair_rows = [], []
    for s1_id in all_s1_ids:
        cand_ids = candidates.get(s1_id, set())
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cand_ids)),
        })
        s1_rec = s1_lookup[s1_id]
        for cand_id in cand_ids:
            cand_rec = lookup.get(cand_id)
            if cand_rec is None:
                continue
            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "name_a": s1_rec.business_name,
                "name_b": cand_rec.business_name,
                "addr_a": s1_rec.business_address,
                "addr_b": cand_rec.business_address,
                "country_a": s1_rec.country,
                "country_b": cand_rec.country,
            })

    candidate_out = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")
    pd.DataFrame(cand_rows).to_csv(candidate_out, sep="\t", index=False)
    print(f"Wrote {len(cand_rows):,} rows to {candidate_out}")

    # --- 2. matching_results.tsv (model-scored, thresholded matches) ---
    if pair_rows:
        pairs_df = pd.DataFrame(pair_rows)
        print(f"Extracting features for {len(pairs_df):,} test pairs...")
        feats = build_feature_matrix(pairs_df)

        print(f"Scoring test pairs with trained model from {config.MODEL_PATH}...")
        model = lgb.Booster(model_file=config.MODEL_PATH)
        pairs_df["score"] = model.predict(feats)

        print(f"Applying decision threshold: {threshold:.2f}...")
        matches = pairs_df[pairs_df["score"] >= threshold]
    else:
        matches = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    grouped = (
        matches.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .to_dict()
    )

    match_rows = [
        {"source1_entity_id": s1_id, "matched_entity_ids": grouped.get(s1_id, "")}
        for s1_id in all_s1_ids
    ]

    matching_out = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")
    pd.DataFrame(match_rows).to_csv(matching_out, sep="\t", index=False)
    print(f"Wrote {len(match_rows):,} rows to {matching_out}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

