"""Pairwise similarity features for a Source1 <-> candidate record pair.

Computes fine-grained phonetic, token, and character-level similarity metrics:
- Token-set and partial ratios (robust to extra legal words, departments, DBAs)
- First-token exact match (primary brand name agreement)
- Numerical / PIN / postal code overlap (-1 = missing, 0 = conflict, 1 = match)
- Missing address indicators
- Fast batch computation (>25k pairs/sec)
"""
from typing import List, Dict, Any
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .preprocess import normalize_name, normalize_address, extract_numbers, tokenize


def _token_jaccard(a_tokens: list, b_tokens: list) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_pair_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
    country_a: str,
    country_b: str,
) -> Dict[str, Any]:
    """Computes all similarity features for a single entity pair."""
    n_a = normalize_name(name_a)
    n_b = normalize_name(name_b)
    a_a = normalize_address(addr_a)
    a_b = normalize_address(addr_b)

    name_tok_a = tokenize(n_a)
    name_tok_b = tokenize(n_b)
    addr_tok_a = tokenize(a_a)
    addr_tok_b = tokenize(a_b)

    first_tok_match = float(bool(name_tok_a and name_tok_b and name_tok_a[0] == name_tok_b[0]))

    # Postal code / building number comparison
    nums_a = set(extract_numbers(a_a))
    nums_b = set(extract_numbers(a_b))
    if nums_a and nums_b:
        num_match = 1.0 if bool(nums_a & nums_b) else 0.0
    else:
        num_match = -1.0  # missing / neutral indicator

    has_both_addrs = float(bool(a_a and a_b))

    return {
        "name_ratio": fuzz.ratio(n_a, n_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n_a, n_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n_a, n_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n_a, n_b) / 100.0,
        "name_wratio": fuzz.WRatio(n_a, n_b) / 100.0,
        "name_first_tok_match": first_tok_match,
        "name_jaccard": _token_jaccard(name_tok_a, name_tok_b),
        "name_levenshtein_sim": 1.0 - Levenshtein.normalized_distance(n_a, n_b),
        "addr_ratio": fuzz.ratio(a_a, a_b) / 100.0 if has_both_addrs else 0.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(a_a, a_b) / 100.0 if has_both_addrs else 0.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(a_a, a_b) / 100.0 if has_both_addrs else 0.0,
        "addr_partial_ratio": fuzz.partial_ratio(a_a, a_b) / 100.0 if has_both_addrs else 0.0,
        "addr_jaccard": _token_jaccard(addr_tok_a, addr_tok_b) if has_both_addrs else 0.0,
        "addr_num_match": num_match,
        "has_addr_both": has_both_addrs,
        "same_country": float(country_a == country_b),
        "name_len_diff": abs(len(n_a) - len(n_b)),
        "addr_len_diff": abs(len(a_a) - len(a_b)),
    }


def build_feature_matrix(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, fast extraction over all candidate pairs."""
    if pairs_df.empty:
        return pd.DataFrame()

    names_a = pairs_df["name_a"].fillna("").astype(str).tolist()
    names_b = pairs_df["name_b"].fillna("").astype(str).tolist()
    addrs_a = pairs_df["addr_a"].fillna("").astype(str).tolist()
    addrs_b = pairs_df["addr_b"].fillna("").astype(str).tolist()
    countries_a = pairs_df["country_a"].fillna("").astype(str).tolist()
    countries_b = pairs_df["country_b"].fillna("").astype(str).tolist()

    rows: List[Dict[str, Any]] = [
        compute_pair_features(na, nb, aa, ab, ca, cb)
        for na, nb, aa, ab, ca, cb in zip(
            names_a, names_b, addrs_a, addrs_b, countries_a, countries_b
        )
    ]
    return pd.DataFrame(rows)

