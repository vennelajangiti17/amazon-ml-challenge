"""Candidate generation (blocking).

Strategy: block by country (open-set — we just groupby whatever string is
present, so France in the test set works without any code change), then
rank Source2/Source3 records against each Source1 record by TF-IDF
character n-gram cosine similarity on the normalized business name. Keep
the top-K above a similarity floor.

This is the piece that sets your recall ceiling — if you have time, add a
second blocking pass on normalized address and take the UNION of the two
candidate sets, since name noise and address noise don't always co-occur.
"""
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import TOP_K_CANDIDATES, MIN_NAME_SIM


def _vectorize(all_names):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vec.fit_transform(all_names)
    return vec, matrix


def generate_candidates(s1_df, s2_df, s3_df, top_k=TOP_K_CANDIDATES, min_sim=MIN_NAME_SIM):
    """Returns dict: source1_entity_id -> set of candidate entity_ids (S2/S3).

    Expects s1_df/s2_df/s3_df to already have a `_norm_name` column
    (see preprocess.normalize_name) and the original `country`, `entity_id`
    columns.
    """
    candidates = defaultdict(set)

    other = pd.concat(
        [s2_df.assign(_source="S2"), s3_df.assign(_source="S3")],
        ignore_index=True,
    )

    for country, s1_group in s1_df.groupby("country"):
        other_group = other[other["country"] == country]
        if other_group.empty or s1_group.empty:
            continue

        s1_names = s1_group["_norm_name"].tolist()
        other_names = other_group["_norm_name"].tolist()
        all_names = s1_names + other_names

        _, matrix = _vectorize(all_names)
        s1_matrix = matrix[: len(s1_names)]
        other_matrix = matrix[len(s1_names):]

        k = min(top_k, len(other_names))
        if k == 0:
            continue
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(other_matrix)
        distances, indices = nn.kneighbors(s1_matrix)

        other_ids = other_group["entity_id"].tolist()
        s1_ids = s1_group["entity_id"].tolist()

        for row_i, s1_id in enumerate(s1_ids):
            for dist, col_j in zip(distances[row_i], indices[row_i]):
                sim = 1 - dist
                if sim >= min_sim:
                    candidates[s1_id].add(other_ids[col_j])

    return candidates
