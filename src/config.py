from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "arXiv_scientific_dataset.csv"
CHUNK_DIR = ROOT / "data" / "chunks"
PROCESSED = ROOT / "data" / "processed.parquet"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ROOT / "models" / "classifier.joblib"

CHUNK_SIZE = 25_000      # rows read from the CSV at a time
MIN_PAPERS = 2_000       # categories smaller than this are dropped
RANDOM_STATE = 42
