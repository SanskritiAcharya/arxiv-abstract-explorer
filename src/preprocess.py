"""Step 1: break the 410 MB CSV into chunks, clean each chunk in parallel,
then combine, de-duplicate and keep only the well-populated categories.

Run: python -m src.preprocess
"""
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.config import ARTIFACTS, CHUNK_DIR, CHUNK_SIZE, MIN_PAPERS, PROCESSED, RAW_CSV
from src.text import clean

COLUMNS = ["id", "title", "category", "category_code", "published_date", "summary", "summary_word_count"]


def clean_chunk(args):
    i, chunk = args
    chunk = chunk.dropna(subset=["summary", "category_code"]).copy()
    chunk["summary"] = chunk["summary"].str.replace(r"\s+", " ", regex=True).str.strip()
    chunk["clean_text"] = chunk["summary"].map(clean)
    out = CHUNK_DIR / f"part-{i:03d}.parquet"
    chunk.to_parquet(out, index=False)
    return i, len(chunk)


def main():
    start = time.time()
    shutil.rmtree(CHUNK_DIR, ignore_errors=True)
    CHUNK_DIR.mkdir(parents=True)
    ARTIFACTS.mkdir(exist_ok=True)

    reader = pd.read_csv(RAW_CSV, usecols=COLUMNS, chunksize=CHUNK_SIZE)
    with ProcessPoolExecutor() as pool:
        for i, n in pool.map(clean_chunk, enumerate(reader)):
            print(f"chunk {i:03d}: {n:,} rows cleaned")

    df = pd.read_parquet(CHUNK_DIR)
    raw_rows = len(df)

    # The same abstract filed under two categories would leak between train and test.
    df = df.drop_duplicates(subset="summary", keep=False)
    df = df[df["clean_text"].str.len() > 0]

    counts = df["category_code"].value_counts()
    kept = counts[counts >= MIN_PAPERS].index
    df = df[df["category_code"].isin(kept)].reset_index(drop=True)
    df["published_date"] = pd.to_datetime(df["published_date"], format="%m/%d/%y")
    df.to_parquet(PROCESSED, index=False)

    names = df.drop_duplicates("category_code").set_index("category_code")["category"].to_dict()
    report = {
        "raw_rows": raw_rows,
        "rows_after_cleaning": len(df),
        "categories_total": int(len(counts)),
        "categories_kept": int(len(kept)),
        "min_papers_per_category": MIN_PAPERS,
        "seconds": round(time.time() - start, 1),
    }
    (ARTIFACTS / "category_names.json").write_text(json.dumps(names, indent=2))
    (ARTIFACTS / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
