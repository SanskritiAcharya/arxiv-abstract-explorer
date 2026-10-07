"""Step 2: measure how abstract text relates to category.

Produces three views of the category-summary relationship:
  1. chi-square: which words are statistically tied to each category
  2. category similarity: cosine similarity of each category's average TF-IDF vector
  3. abstract length by category, with eta-squared (share of length variance
     explained by category)

Run: python -m src.analyze
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ARTIFACTS, PROCESSED

TOP_N = 15


def top_terms_by_chi2(df):
    vec = CountVectorizer(binary=True, min_df=20, max_df=0.5, tokenizer=str.split,
                          lowercase=False, token_pattern=None)
    X = vec.fit_transform(df["clean_text"])
    vocab = np.array(vec.get_feature_names_out())
    result = {}
    for code in sorted(df["category_code"].unique()):
        is_cat = (df["category_code"] == code).to_numpy()
        scores, _ = chi2(X, is_cat)
        # chi-square is two-sided; keep only words over-represented in this category
        in_cat = np.asarray(X[is_cat].mean(axis=0)).ravel()
        out_cat = np.asarray(X[~is_cat].mean(axis=0)).ravel()
        scores = np.where(in_cat > out_cat, np.nan_to_num(scores), 0)
        best = np.argsort(scores)[::-1][:TOP_N]
        result[code] = [{"word": vocab[j], "chi2": round(float(scores[j]), 1),
                         "pct_in_category": round(100 * float(in_cat[j]), 1),
                         "pct_elsewhere": round(100 * float(out_cat[j]), 1)} for j in best]
    return result


def category_similarity(df):
    vec = TfidfVectorizer(min_df=20, max_df=0.5, max_features=50_000, sublinear_tf=True,
                          tokenizer=str.split, lowercase=False, token_pattern=None)
    X = vec.fit_transform(df["clean_text"])
    codes = sorted(df["category_code"].unique())
    centroids = np.vstack([np.asarray(X[(df["category_code"] == c).to_numpy()].mean(axis=0))
                           for c in codes])
    sim = cosine_similarity(centroids)
    return {"codes": codes, "matrix": np.round(sim, 3).tolist()}


def length_by_category(df):
    words = df["summary_word_count"]
    stats = df.groupby("category_code")["summary_word_count"].agg(["mean", "median", "std", "count"])
    grand = words.mean()
    ss_between = (stats["count"] * (stats["mean"] - grand) ** 2).sum()
    ss_total = ((words - grand) ** 2).sum()
    return {"eta_squared": round(float(ss_between / ss_total), 4),
            "by_category": stats.round(1).reset_index().to_dict(orient="records")}


def main():
    df = pd.read_parquet(PROCESSED)
    counts = df["category_code"].value_counts()
    yearly = (df.assign(year=df["published_date"].dt.year)
                .groupby(["year", "category_code"]).size().rename("papers").reset_index())

    out = {
        "category_counts": counts.to_dict(),
        "papers_per_year": yearly.to_dict(orient="records"),
        "top_terms": top_terms_by_chi2(df),
        "similarity": category_similarity(df),
        "length": length_by_category(df),
    }
    (ARTIFACTS / "analysis.json").write_text(json.dumps(out, indent=2, default=int))
    print("eta^2 (length ~ category):", out["length"]["eta_squared"])
    for code, terms in out["top_terms"].items():
        print(f"{code:8s}", ", ".join(t["word"] for t in terms[:8]))


if __name__ == "__main__":
    main()
