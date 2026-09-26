# Business Entity Resolution — Amazon ML Challenge 2026

Matches business records across Source 1 (reference), Source 2, and Source 3
using TF-IDF blocking + a LightGBM pairwise classifier.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Expected data layout

Unzip the challenge dataset so it looks like this (relative to this folder,
or point `DATA_DIR` at wherever you put it):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run

```bash
python -m src.train      # trains model.txt from dataset/train/
python -m src.predict    # writes output/matching_results.tsv + output/candidate_pairs.tsv from dataset/test/
```

Override paths via env vars if your layout differs:
`DATA_DIR`, `OUTPUT_DIR`, `MODEL_PATH` (see `src/config.py`).

Then validate before uploading:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```//run the validator provided in the challenge's student_resource/ folder

## How it works

1. **`preprocess.py`** — normalizes names (strips legal suffixes, punctuation)
   and addresses (expands abbreviations like Rd→Road).
2. **`blocking.py`** — for each Source1 entity, finds the top-K most
   TF-IDF-similar Source2/Source3 records **within the same country**, above
   a similarity floor. Country is treated as an open string label (no
   hardcoded set), so it works unchanged on France in the test set.
3. **`features.py`** — computes ~10 similarity features per candidate pair
   (Levenshtein/Jaccard/token-sort ratios on name and address, country match,
   length diffs).
4. **`train.py`** — builds positive pairs from ground truth + negative pairs
   from blocking candidates that *aren't* true matches, trains a LightGBM
   binary classifier.
5. **`predict.py`** — reruns blocking on test data (→ `candidate_pairs.tsv`),
   scores every candidate with the trained model, keeps those above
   `MATCH_THRESHOLD`, groups by Source1 entity (→ `matching_results.tsv`).
6. **`evaluate.py`** — macro F_0.5 scorer matching the challenge's own
   formula, for tuning the threshold on a held-out split.

## Where to spend your remaining time (roughly in priority order)

- **Blocking recall.** Run `evaluate.py`-style checks on your candidate sets
  alone (does the true match ever appear in `candidates[s1_id]`?) before
  worrying about the classifier — recall lost here can't be recovered later.
  Consider adding a second blocking pass on normalized address and taking
  the union with the name-based candidates.
- **Threshold tuning.** F_0.5 weights precision 2x. Sweep `MATCH_THRESHOLD`
  against a held-out slice of `train_ground_truth.tsv` using
  `evaluate.macro_f_beta` and pick the value that actually maximizes it —
  don't just use 0.5.
- **Singletons.** A large fraction of Source1 entities may have zero true
  matches. Getting the "no match" call right is worth a full 1.0 per entity —
  make sure your threshold isn't too permissive.
- **Feature set.** Add phonetic features (e.g. Soundex/Metaphone) and
  address-component parsing (city/state/PIN extraction) if you have time —
  the current feature set is intentionally minimal to get something running
  end-to-end fast.
- **License/compliance.** LightGBM is BSD/MIT-family licensed and nowhere
  near the 8B parameter cap, and nothing here calls out to external
  services — keep it that way per the fair-play rules.

## Team workflow

This folder is meant to be pushed as-is to a shared GitHub repo so everyone
can branch/PR independently (e.g. one person tunes blocking, another
iterates on features, another on the model). See the root-level `dataset/`
and `output/` folders are gitignored — only code is versioned; each
teammate keeps their own local copy of the (large) dataset.
