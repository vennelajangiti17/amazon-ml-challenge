"""Central config: paths and tunable knobs for the entity resolution pipeline."""
import os

DATA_DIR = os.environ.get("DATA_DIR", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.txt")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

# Blocking / candidate generation
TOP_K_CANDIDATES = 15   # candidates per Source1 entity, per country block
MIN_NAME_SIM = 0.30     # floor TF-IDF cosine similarity to keep a candidate

# Matching
MATCH_THRESHOLD = 0.5   # probability threshold — retune on your validation split
RANDOM_STATE = 42
