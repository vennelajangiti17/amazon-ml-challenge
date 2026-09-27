# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Insert Team Name]  
**Team Members:** Vennela Jangiti, [Teammate 2], [Teammate 3], [Teammate 4]  
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We developed a highly scalable, multi-lingual, and precision-optimized Entity Resolution pipeline combining country-partitioned TF-IDF sparse blocking with a gradient-boosted pairwise classifier (LightGBM). By engineering fine-grained token-set, character n-gram, and numerical landmark features, and tuning the decision threshold specifically against the competition's macro-averaged $F_{0.5}$ metric, our system achieves high recall while strongly suppressing false positive merges.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory analysis across the training and test sets (covering US, India, and France), we identified several key challenges:
- **Severe Name Noise & Abbreviations**: High variance in legal suffixes (`pvt ltd`, `inc`, `sarl`, `sasu`, `et fils`), abbreviations, and URL domain artifacts (`maurewilliamscolombier.com` matching corporate names).
- **Accented Characters**: The test set introduces France as an open-set country with diacritics (`é`, `ô`, `à`, `ç`) that would cause false negatives without Unicode NFKD decomposition.
- **Missing Address Fields**: Many candidate records in Source 2 and Source 3 lack addresses (`NaN`), requiring the model to adaptively rely on name similarity when address data is absent.
- **Landmark-Based & Non-Standard Addresses**: Indian and French addresses frequently contain landmark references (`near SBI`, `opp bus stand`, `b/h Natraj cinema`) and localized abbreviations (`bd`, `av`, `r`, `flt no`, `sec`).
- **Precision Penalty in Evaluation Metric**: The competition evaluates via macro-averaged $F_{0.5}$, weighting precision 2× over recall and penalizing false merges on singletons heavily.

### 2.2 Solution Strategy
- **Approach Type**: Country-Partitioned Sparse Blocking + Gradient Boosted Pairwise Classifier (LightGBM).
- **Core Innovations**:
  1. Fast sparse matrix dot-product candidate generation partitioned by country with adaptive stop-word filtering.
  2. Multi-lingual Unicode normalization and standardized legal entity stripping across US, India, and France.
  3. Fine-grained numeric token and landmark overlap extraction (`addr_num_match`).
  4. Decision threshold optimization tuned specifically to maximize the precision-heavy macro $F_{0.5}$ score.

---

## 3. Candidate Generation (Blocking)
To reduce the $O(N \times M)$ comparison space across 1.73M queries and 10.3M candidate records:
- **Blocking keys used**: Country partitioning (strictly open-set, allowing seamless handling of France) combined with TF-IDF Word unigrams (`max_features=50,000`, `min_df=2`, `sublinear_tf=True`).
- **Candidate pairs generated**: 25,926,310 total candidate pairs across the test set (~14.9 candidates per query).
- **How true matches were preserved**:
  - We vectorized queries and candidates independently per country block, capturing sub-token brand agreements while discarding ubiquitous stop-words.
  - Verification on training ground-truth confirmed an **86.5% recall ceiling** on candidate generation alone.

---

## 4. Matching Model

**Features used (18 total):**
- **Name features**: Rapidfuzz `fuzz.ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `WRatio`, first-token exact match indicator, token Jaccard similarity, normalized Levenshtein similarity, character length difference.
- **Address features**: Rapidfuzz `token_set_ratio`, `partial_ratio`, `token_sort_ratio`, `ratio`, token Jaccard similarity, character length difference, missing address indicator (`has_addr_both`).
- **Numerical & Geo features**: `addr_num_match` (compares extracted building numbers, 6-digit Indian PIN codes, and 5-digit US/French postal codes: +1.0 for match, 0.0 for conflict, -1.0 for neutral/missing).
- **Country match**: Open-set exact equality indicator.

**Model type**: LightGBM Binary Classifier (400 trees, `learning_rate=0.08`, `num_leaves=63`, `min_child_samples=20`, trained with early stopping on validation binary log-loss).

**Threshold selection method**:
- Macro $F_{0.5}$ score was computed across probability thresholds from 0.30 to 0.90 in increments of 0.05 on an unseen validation split of 15,000 Source 1 entities.
- While default 0.50 threshold scored 0.944, shifting the threshold to **`0.65`** achieved the peak validation score of **0.95438**, as higher precision significantly reduces false merges on singletons.

---

## 5. Results & Error Analysis

- **Best Validation Macro $F_{0.5}$ Score**: **0.95438** (Validation log-loss: 0.0437).
- **Top Feature Contributors**:
  1. `addr_jaccard` (1917 splits)
  2. `addr_partial_ratio` (1813 splits)
  3. `addr_token_set_ratio` (1625 splits)
  4. `addr_len_diff` (1561 splits)
  5. `addr_token_sort_ratio` (1541 splits)
  6. `addr_ratio` (1337 splits)
  7. `name_partial_ratio` (1300 splits)
  8. `name_len_diff` (1232 splits)
  9. `name_ratio` (1157 splits)
  10. `name_levenshtein_sim` (1025 splits)
- **Common False Positives (Suppressed)**: Different businesses located in the same corporate park or commercial building with similar trade suffixes. Suppressed by combining first-token brand matching with address number checking.
- **Common False Negatives**: Businesses where the legal reference name and the registered trade name share no lexical overlap and address information is completely omitted (`NaN`).

---

## 6. Conclusion
Our solution demonstrates that combining robust multi-lingual text normalization with country-partitioned sparse blocking and precision-focused feature engineering yields a state-of-the-art entity resolution pipeline. By evaluating strictly on unseen validation businesses and optimizing for macro $F_{0.5}$, the pipeline delivers high scalability and exceptional precision across US, India, and France datasets.

---

## Appendix

### A. Code Artefacts
- **`src/preprocess.py`**: Multi-lingual text cleaner, accent normalizer, legal suffix stripper, and number extractor.
- **`src/blocking.py`**: Country-partitioned TF-IDF sparse matrix candidate generator.
- **`src/features.py`**: High-speed vectorized feature extraction engine (>25k pairs/sec).
- **`src/train.py`**: LightGBM training script with automatic validation threshold sweeping.
- **`src/predict.py`**: End-to-end streaming inference writing `candidate_pairs.tsv` and `matching_results.tsv`.
- **`utils/validate_submission.py`**: Submission verification script.

