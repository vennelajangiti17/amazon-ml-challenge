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

    candidate_out = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")
    matching_out = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")

    print(f"Streaming candidate pairs and scoring with threshold {threshold:.2f}...")
    model = lgb.Booster(model_file=config.MODEL_PATH)
    matches_dict = {}

    BATCH_SIZE = 150000
    current_batch = []
    total_pairs_scored = 0

    with open(candidate_out, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, s1_id in enumerate(all_s1_ids):
            cand_ids = candidates.get(s1_id, set())
            f_cand.write(f"{s1_id}\t{','.join(sorted(cand_ids))}\n")

            s1_rec = s1_lookup.get(s1_id)
            if s1_rec is None:
                continue

            for cand_id in cand_ids:
                cand_rec = lookup.get(cand_id)
                if cand_rec is None:
                    continue
                current_batch.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "name_a": s1_rec.business_name,
                    "name_b": cand_rec.business_name,
                    "addr_a": s1_rec.business_address,
                    "addr_b": cand_rec.business_address,
                    "country_a": s1_rec.country,
                    "country_b": cand_rec.country,
                })

                if len(current_batch) >= BATCH_SIZE:
                    batch_df = pd.DataFrame(current_batch)
                    batch_feats = build_feature_matrix(batch_df)
                    scores = model.predict(batch_feats)
                    for s1_i, cand_i, sc in zip(
                        batch_df["source1_entity_id"], batch_df["candidate_entity_id"], scores
                    ):
                        if sc >= threshold:
                            if s1_i not in matches_dict:
                                matches_dict[s1_i] = set()
                            matches_dict[s1_i].add(cand_i)
                    total_pairs_scored += len(current_batch)
                    print(f"  Scored {total_pairs_scored:,} candidate pairs...")
                    current_batch = []

        # Flush any remaining pairs in the final batch
        if current_batch:
            batch_df = pd.DataFrame(current_batch)
            batch_feats = build_feature_matrix(batch_df)
            scores = model.predict(batch_feats)
            for s1_i, cand_i, sc in zip(
                batch_df["source1_entity_id"], batch_df["candidate_entity_id"], scores
            ):
                if sc >= threshold:
                    if s1_i not in matches_dict:
                        matches_dict[s1_i] = set()
                    matches_dict[s1_i].add(cand_i)
            total_pairs_scored += len(current_batch)
            print(f"  Scored {total_pairs_scored:,} candidate pairs (complete)...")
            current_batch = []

    print(f"Wrote {len(all_s1_ids):,} rows to {candidate_out}")

    print(f"Writing {len(all_s1_ids):,} final prediction rows to {matching_out}...")
    with open(matching_out, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            m_set = matches_dict.get(s1_id, set())
            m_str = ",".join(sorted(m_set)) if m_set else ""
            f_match.write(f"{s1_id}\t{m_str}\n")

    print(f"Wrote {len(all_s1_ids):,} rows to {matching_out}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()


