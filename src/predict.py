"""Run inference on the test set and write the two required output files.

Run from the `code/business_entity_resolution` directory:
    python -m src.predict
"""
import os
import pandas as pd
import lightgbm as lgb

from . import config
from .preprocess import normalize_name
from .blocking import generate_candidates
from .features import build_feature_matrix


def load_test_sources():
    s1 = pd.read_csv(config.TEST_S1, sep="\t")
    s2 = pd.read_csv(config.TEST_S2, sep="\t")
    s3 = pd.read_csv(config.TEST_S3, sep="\t")
    for df in (s1, s2, s3):
        df["_norm_name"] = df["business_name"].apply(normalize_name)
    return s1, s2, s3


def main(threshold: float = config.MATCH_THRESHOLD):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    s1, s2, s3 = load_test_sources()
    candidates = generate_candidates(s1, s2, s3)

    lookup = {r["entity_id"]: r for _, r in pd.concat([s2, s3], ignore_index=True).iterrows()}
    s1_lookup = {r["entity_id"]: r for _, r in s1.iterrows()}
    all_s1_ids = list(s1["entity_id"])

    # --- candidate_pairs.tsv (blocking output, one row per Source1 entity) ---
    cand_rows, pair_rows = [], []
    for s1_id in all_s1_ids:
        cand_ids = candidates.get(s1_id, set())
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cand_ids)),
        })
        s1_rec = s1_lookup[s1_id]
        for cand_id in cand_ids:
            cand_rec = lookup[cand_id]
            pair_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "name_a": s1_rec["business_name"], "name_b": cand_rec["business_name"],
                "addr_a": s1_rec["business_address"], "addr_b": cand_rec["business_address"],
                "country_a": s1_rec["country"], "country_b": cand_rec["country"],
            })

    pd.DataFrame(cand_rows).to_csv(
        os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False
    )

    # --- matching_results.tsv (model-scored, thresholded matches) ---
    if pair_rows:
        pairs_df = pd.DataFrame(pair_rows)
        feats = build_feature_matrix(pairs_df)
        model = lgb.Booster(model_file=config.MODEL_PATH)
        pairs_df["score"] = model.predict(feats)
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
    pd.DataFrame(match_rows).to_csv(
        os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False
    )
    print("Wrote output/matching_results.tsv and output/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
