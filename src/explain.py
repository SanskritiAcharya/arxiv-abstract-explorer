"""Word highlights: how much each word in an abstract pushed the model
towards (or away from) a category.

For a linear model the score for class c is sum_j x_j * w_cj, so each feature's
contribution is x_j * w_cj. Single-word features go to that word; two-word
phrase features are split equally between their two words.
"""
import numpy as np

from src.text import tokens_with_spans


def predict(bundle, text, top_k=3):
    vec, model = bundle["vectorizer"], bundle["model"]
    spans = list(tokens_with_spans(text))
    kept = [s for s in spans if s[3]]
    x = vec.transform([" ".join(s[2] for s in kept)])

    probs = model.predict_proba(x)[0]
    order = np.argsort(probs)[::-1][:top_k]
    predictions = [{"code": model.classes_[i], "probability": round(float(probs[i]), 4)} for i in order]

    best = order[0]
    weights = model.coef_[best]
    vocab = vec.vocabulary_
    row = x.tocsr()
    tfidf = dict(zip(row.indices, row.data))

    def contribution(term):
        j = vocab.get(term)
        return tfidf.get(j, 0.0) * weights[j] if j is not None else 0.0

    # score each kept token (index into `kept`)
    scores = np.zeros(len(kept))
    for i, (_, _, lem, _) in enumerate(kept):
        scores[i] += contribution(lem)
        if i + 1 < len(kept):
            share = contribution(f"{lem} {kept[i + 1][2]}") / 2
            scores[i] += share
            scores[i + 1] += share

    # rebuild the original text as segments so the frontend can colour it
    segments, cursor = [], 0
    score_at = {s[0]: scores[i] for i, s in enumerate(kept)}
    for start, end, _, _ in spans:
        if start > cursor:
            segments.append({"text": text[cursor:start], "score": 0.0})
        segments.append({"text": text[start:end], "score": round(float(score_at.get(start, 0.0)), 4)})
        cursor = end
    if cursor < len(text):
        segments.append({"text": text[cursor:], "score": 0.0})

    # strongest words for the predicted category, merged across repeats
    totals = {}
    for (_, _, lem, _), s in zip(kept, scores):
        totals[lem] = totals.get(lem, 0.0) + s
    top_words = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "predictions": predictions,
        "segments": segments,
        "top_words": [{"word": w, "score": round(float(s), 4)} for w, s in top_words if s > 0],
    }
