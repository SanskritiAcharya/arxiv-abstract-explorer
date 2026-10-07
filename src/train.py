"""Step 3: train the category classifier and evaluate it.

Baseline: Complement Naive Bayes. Main model: Logistic Regression.
Both use TF-IDF features (single words + two-word phrases).

Run: python -m src.train
"""
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

from src.config import ARTIFACTS, MODEL_PATH, PROCESSED, RANDOM_STATE


def evaluate(name, model, X_test, y_test, labels):
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, pred, average="weighted"), 4),
        "per_class": classification_report(y_test, pred, labels=labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }


def main():
    df = pd.read_parquet(PROCESSED)
    train, test = train_test_split(df, test_size=0.2, stratify=df["category_code"],
                                   random_state=RANDOM_STATE)
    labels = sorted(df["category_code"].unique())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=150_000,
                                 sublinear_tf=True, tokenizer=str.split, lowercase=False,
                                 token_pattern=None)
    t = time.time()
    X_train = vectorizer.fit_transform(train["clean_text"])
    X_test = vectorizer.transform(test["clean_text"])
    print(f"TF-IDF: {X_train.shape[1]:,} features in {time.time() - t:.0f}s")

    results = []
    t = time.time()
    nb = ComplementNB(alpha=0.3).fit(X_train, train["category_code"])
    results.append(evaluate("Complement Naive Bayes (baseline)", nb, X_test, test["category_code"], labels))
    print(f"Naive Bayes trained in {time.time() - t:.0f}s")

    t = time.time()
    # class_weight="balanced" stops the huge cs.LG / cs.CV classes from drowning out small ones
    lr = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
    lr.fit(X_train, train["category_code"])
    results.append(evaluate("Logistic Regression", lr, X_test, test["category_code"], labels))
    print(f"Logistic Regression trained in {time.time() - t:.0f}s")

    for r in results:
        print(f"{r['model']:36s} accuracy={r['accuracy']:.3f}  macro F1={r['macro_f1']:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": lr}, MODEL_PATH)

    metrics = {"labels": labels, "train_size": len(train), "test_size": len(test),
               "n_features": X_train.shape[1], "results": results}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Held-out abstracts for the app's "try a random paper" button.
    samples = (test.groupby("category_code").sample(n=20, random_state=RANDOM_STATE)
               [["title", "category_code", "summary"]])
    samples.to_json(ARTIFACTS / "samples.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
