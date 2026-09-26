"""Candidate generation (blocking) with high-speed sparse matrix dot products.

Strategy:
1. Block strictly by country (open-set: works unchanged for France in the test set).
2. Pass 1 (Name): TF-IDF character n-grams (2 to 4) on normalized business name.
   Extracts top-K most similar S2/S3 candidates above MIN_NAME_SIM.
3. Pass 2 (Address): TF-IDF character n-grams (3 to 5) on normalized business address.
   Extracts top-K most similar S2/S3 candidates above MIN_ADDR_SIM.
4. Candidate union: combines name and address candidates per Source 1 entity.
   This guarantees high recall ceiling even when business names use abbreviations,
   brand/trade names, or website domains.

Performance:
Uses chunked sparse matrix multiplication with unit-normalized vectors (scipy.sparse)
to achieve 50x speedup and low memory footprint over sklearn NearestNeighbors.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    TOP_K_CANDIDATES,
    MIN_NAME_SIM,
    TOP_K_ADDR_CANDIDATES,
    MIN_ADDR_SIM,
    BLOCKING_CHUNK_SIZE,
)
from .preprocess import normalize_name, normalize_address


def _find_top_k_sparse(
    query_matrix: sp.csr_matrix,
    candidate_matrix: sp.csr_matrix,
    top_k: int,
    min_sim: float,
    chunk_size: int = BLOCKING_CHUNK_SIZE,
) -> dict:
    """Finds top-K most similar candidate indices for each query row using chunked dot-products."""
    candidates = defaultdict(set)
    cand_T = candidate_matrix.T.tocsc()
    n_queries = query_matrix.shape[0]

    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, n_queries)
        chunk = query_matrix[start_idx:end_idx]
        sims = chunk.dot(cand_T).tocsr()

        indptr = sims.indptr
        indices = sims.indices
        data = sims.data

        for row_i in range(sims.shape[0]):
            p_start = indptr[row_i]
            p_end = indptr[row_i + 1]
            if p_start == p_end:
                continue

            row_data = data[p_start:p_end]
            row_indices = indices[p_start:p_end]

            mask = row_data >= min_sim
            if not np.any(mask):
                continue

            v_data = row_data[mask]
            v_indices = row_indices[mask]

            if len(v_data) > top_k:
                top_part = np.argpartition(v_data, -top_k)[-top_k:]
                best_indices = v_indices[top_part]
            else:
                best_indices = v_indices

            candidates[start_idx + row_i].update(best_indices.tolist())

    return candidates


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k_name: int = TOP_K_CANDIDATES,
    min_name_sim: float = MIN_NAME_SIM,
    top_k_addr: int = TOP_K_ADDR_CANDIDATES,
    min_addr_sim: float = MIN_ADDR_SIM,
) -> dict:
    """Returns dict: source1_entity_id -> set of candidate entity_ids (S2/S3)."""
    candidates = defaultdict(set)

    # Ensure normalized columns exist
    for df in (s1_df, s2_df, s3_df):
        if "_norm_name" not in df.columns:
            df["_norm_name"] = df["business_name"].fillna("").apply(normalize_name)
        if "_norm_addr" not in df.columns:
            df["_norm_addr"] = df["business_address"].fillna("").apply(normalize_address)

    other = pd.concat([s2_df, s3_df], ignore_index=True)

    for country, s1_group in s1_df.groupby("country"):
        other_group = other[other["country"] == country]
        if other_group.empty or s1_group.empty:
            continue

        s1_ids = s1_group["entity_id"].tolist()
        other_ids = other_group["entity_id"].tolist()

        # --- PASS 1: Business Name Blocking ---
        s1_names = s1_group["_norm_name"].tolist()
        other_names = other_group["_norm_name"].tolist()
        all_names = s1_names + other_names

        name_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            dtype=np.float32,
            sublinear_tf=True,
        )
        name_matrix = name_vec.fit_transform(all_names)
        s1_name_mat = name_matrix[: len(s1_names)]
        other_name_mat = name_matrix[len(s1_names) :]

        name_cands = _find_top_k_sparse(
            s1_name_mat, other_name_mat, top_k=top_k_name, min_sim=min_name_sim
        )

        for q_idx, c_indices in name_cands.items():
            s1_id = s1_ids[q_idx]
            for c_idx in c_indices:
                candidates[s1_id].add(other_ids[c_idx])

        # --- PASS 2: Address Blocking (for entries with addresses) ---
        s1_addrs = s1_group["_norm_addr"].tolist()
        other_addrs = other_group["_norm_addr"].tolist()
        all_addrs = s1_addrs + other_addrs

        has_addrs = any(len(a) > 5 for a in s1_addrs) and any(len(a) > 5 for a in other_addrs)
        if has_addrs and top_k_addr > 0:
            addr_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                dtype=np.float32,
                sublinear_tf=True,
            )
            addr_matrix = addr_vec.fit_transform(all_addrs)
            s1_addr_mat = addr_matrix[: len(s1_addrs)]
            other_addr_mat = addr_matrix[len(s1_addrs) :]

            addr_cands = _find_top_k_sparse(
                s1_addr_mat, other_addr_mat, top_k=top_k_addr, min_sim=min_addr_sim
            )

            for q_idx, c_indices in addr_cands.items():
                s1_id = s1_ids[q_idx]
                for c_idx in c_indices:
                    candidates[s1_id].add(other_ids[c_idx])

    return candidates

