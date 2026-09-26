"""Pairwise similarity features for a Source1 <-> candidate record pair."""
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .preprocess import normalize_name, normalize_address, tokenize


def _token_jaccard(a_tokens, b_tokens) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(row) -> dict:
    """row needs: name_a, name_b, addr_a, addr_b, country_a, country_b (raw text)."""
    name_a, name_b = normalize_name(row["name_a"]), normalize_name(row["name_b"])
    addr_a, addr_b = normalize_address(row["addr_a"]), normalize_address(row["addr_b"])

    name_tok_a, name_tok_b = tokenize(name_a), tokenize(name_b)
    addr_tok_a, addr_tok_b = tokenize(addr_a), tokenize(addr_b)

    return {
        "name_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_jaccard": _token_jaccard(name_tok_a, name_tok_b),
        "name_levenshtein_sim": 1 - Levenshtein.normalized_distance(name_a, name_b),
        "addr_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr_a, addr_b) / 100.0,
        "addr_jaccard": _token_jaccard(addr_tok_a, addr_tok_b),
        "same_country": float(row["country_a"] == row["country_b"]),
        "name_len_diff": abs(len(name_a) - len(name_b)),
        "addr_len_diff": abs(len(addr_a) - len(addr_b)),
    }


def build_feature_matrix(pairs_df: pd.DataFrame) -> pd.DataFrame:
    return pairs_df.apply(pair_features, axis=1, result_type="expand")
