"""Streamlit frontend. Talks to the FastAPI backend over HTTP.

Run (with the API already running): streamlit run app/streamlit_app.py
"""
import html
import os
import sys
import threading
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="arXiv Abstract Explorer", page_icon="📄", layout="wide")


@st.cache_resource
def start_embedded_api():
    """On hosts that run only Streamlit (e.g. Streamlit Community Cloud), start the
    FastAPI backend in a background thread inside the same container."""
    import uvicorn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        if server.started:
            break
        time.sleep(0.5)
    return "http://127.0.0.1:8000"


# Set API_URL to use a separately running backend; otherwise one is started here.
API = os.environ.get("API_URL") or start_embedded_api()


@st.cache_data(ttl=600)
def get(path):
    r = requests.get(f"{API}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def highlight_html(segments):
    """Green = pushed towards the predicted category, red = pushed away."""
    peak = max((abs(s["score"]) for s in segments), default=0) or 1
    parts = []
    for seg in segments:
        text = html.escape(seg["text"])
        strength = abs(seg["score"]) / peak
        if strength < 0.05:
            parts.append(text)
            continue
        colour = "22, 163, 74" if seg["score"] > 0 else "220, 38, 38"
        alpha = 0.15 + 0.65 * strength
        parts.append(f'<span title="{seg["score"]:+.3f}" style="background: rgba({colour}, {alpha:.2f});'
                     f' border-radius: 3px; padding: 0 2px;">{text}</span>')
    return f'<div style="line-height: 1.9; font-size: 1.02rem;">{"".join(parts)}</div>'


try:
    names = get("/categories")
except requests.RequestException:
    st.error(f"Can't reach the API at {API}. Start it with `uvicorn api.main:app` and refresh.")
    st.stop()

label = lambda code: f"{code} · {names.get(code, code)}"

st.title("arXiv Abstract Explorer")
st.caption("NLP on 260,000 arXiv abstracts: what an abstract's words reveal about its research field.")
tab_classify, tab_explore, tab_model = st.tabs(["🔍 Classify an abstract", "📊 Category vs. summary",
                                                "🎯 Model performance"])

# ---------------------------------------------------------------- classify
with tab_classify:
    if "abstract" not in st.session_state:
        st.session_state.abstract, st.session_state.truth = "", None

    left, right = st.columns([3, 1])
    with right:
        pick = st.selectbox("Load a held-out paper from", ["any category", *sorted(names)],
                            format_func=lambda c: c if c == "any category" else label(c))
        if st.button("🎲 Random unseen paper", width="stretch"):
            params = "" if pick == "any category" else f"?category={pick}"
            s = requests.get(f"{API}/sample{params}", timeout=30).json()
            st.session_state.abstract = s["summary"]
            st.session_state.truth = (s["title"], s["category_code"])
    with left:
        text = st.text_area("Paste an abstract", key="abstract", height=200,
                            placeholder="Paste a research abstract here…")

    if st.session_state.truth:
        title, code = st.session_state.truth
        st.caption(f"**{title}**, actual category: **{label(code)}**")

    if st.button("Classify", type="primary", disabled=len(text.strip()) < 30):
        r = requests.post(f"{API}/predict", json={"text": text, "top_k": 3}, timeout=30)
        if r.status_code != 200:
            st.warning(r.json().get("detail", "Prediction failed."))
        else:
            res = r.json()
            best = res["predictions"][0]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted category", best["code"], f"{best['probability']:.0%} confidence",
                          delta_color="off")
                st.caption(best["name"])
                if st.session_state.truth:
                    ok = best["code"] == st.session_state.truth[1]
                    st.markdown("✅ Correct" if ok else "❌ Model got this one wrong")
                probs = pd.DataFrame(res["predictions"])
                st.altair_chart(alt.Chart(probs).mark_bar().encode(
                    x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("code:N", sort="-x", title=None),
                    tooltip=["name", alt.Tooltip("probability:Q", format=".1%")]), width="stretch")
            with c2:
                st.markdown("**Word highlights**: green words pushed the model towards "
                            f"**{best['code']}**, red words pushed it away.")
                st.markdown(highlight_html(res["segments"]), unsafe_allow_html=True)
                if res["top_words"]:
                    st.markdown("**Strongest evidence:** " +
                                ", ".join(f"`{w['word']}`" for w in res["top_words"]))

# ---------------------------------------------------------------- explore
with tab_explore:
    ov, assoc = get("/stats/overview"), get("/stats/association")
    pre = ov["preprocess"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw papers", f"{pre['raw_rows']:,}")
    m2.metric("After cleaning", f"{pre['rows_after_cleaning']:,}")
    m3.metric("Categories kept", f"{pre['categories_kept']} of {pre['categories_total']}")
    m4.metric("Preprocessing time", f"{pre['seconds']:.0f}s",
              help="CSV read in 25,000-row chunks, cleaned in parallel")

    counts = pd.DataFrame(ov["category_counts"].items(), columns=["code", "papers"])
    counts["name"] = counts["code"].map(names)
    st.subheader("Papers per category")
    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        x="papers:Q", y=alt.Y("code:N", sort="-x", title=None), tooltip=["name", "papers"]),
        width="stretch")
    st.caption("Heavily imbalanced: four AI fields make up most of the data, so the model uses class weighting.")

    st.subheader("Which words define each category? (chi-square test)")
    code = st.selectbox("Category", sorted(assoc["top_terms"]), format_func=label)
    terms = pd.DataFrame(assoc["top_terms"][code])
    st.altair_chart(alt.Chart(terms).mark_bar().encode(
        x=alt.X("chi2:Q", title="chi-square score (higher = stronger link)"),
        y=alt.Y("word:N", sort="-x", title=None),
        tooltip=["word", "chi2", alt.Tooltip("pct_in_category", title="% of papers in category"),
                 alt.Tooltip("pct_elsewhere", title="% of papers elsewhere")]), width="stretch")

    st.subheader("How similar is the language of each category?")
    sim = assoc["similarity"]
    sim_df = pd.DataFrame(sim["matrix"], index=sim["codes"], columns=sim["codes"]).stack().reset_index()
    sim_df.columns = ["a", "b", "similarity"]
    st.altair_chart(alt.Chart(sim_df).mark_rect().encode(
        x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None),
        color=alt.Color("similarity:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["a", "b", "similarity"]).properties(height=450), width="stretch")
    st.caption("Cosine similarity of each category's average TF-IDF vector. Dark off-diagonal cells "
               "(e.g. cs.LG ↔ stat.ML) are fields that write alike, and the model confuses them most.")

    st.subheader("Does abstract length depend on category?")
    length = pd.DataFrame(assoc["length"]["by_category"])
    st.altair_chart(alt.Chart(length).mark_bar().encode(
        x=alt.X("mean:Q", title="average words per abstract"),
        y=alt.Y("category_code:N", sort="-x", title=None), tooltip=["category_code", "mean", "median"]),
        width="stretch")
    st.caption(f"η² = {assoc['length']['eta_squared']:.3f}: category explains only "
               f"{assoc['length']['eta_squared']:.1%} of the variation in length. The link between "
               "category and summary is in *which words* are used, not how many.")

# ---------------------------------------------------------------- model
with tab_model:
    mt = get("/stats/model")
    st.caption(f"Trained on {mt['train_size']:,} abstracts, tested on {mt['test_size']:,} unseen abstracts, "
               f"{mt['n_features']:,} TF-IDF features (words + two-word phrases).")
    summary = pd.DataFrame([{k: r[k] for k in ("model", "accuracy", "macro_f1", "weighted_f1")}
                            for r in mt["results"]])
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Macro F1 averages every category equally, so it exposes weakness on small categories "
               "that plain accuracy hides.")

    best = mt["results"][-1]
    per = pd.DataFrame([{"code": c, **best["per_class"][c]} for c in mt["labels"]])
    st.subheader("F1 score per category (Logistic Regression)")
    st.altair_chart(alt.Chart(per).mark_bar().encode(
        x=alt.X("f1-score:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("code:N", sort="-x", title=None),
        tooltip=["code", alt.Tooltip("precision:Q", format=".2f"), alt.Tooltip("recall:Q", format=".2f"),
                 alt.Tooltip("f1-score:Q", format=".2f"), "support"]), width="stretch")

    st.subheader("Confusion matrix (row-normalised)")
    cm = pd.DataFrame(best["confusion_matrix"], index=mt["labels"], columns=mt["labels"])
    cm = cm.div(cm.sum(axis=1), axis=0).stack().reset_index()
    cm.columns = ["actual", "predicted", "share"]
    st.altair_chart(alt.Chart(cm).mark_rect().encode(
        x="predicted:N", y="actual:N", color=alt.Color("share:Q", scale=alt.Scale(scheme="oranges")),
        tooltip=["actual", "predicted", alt.Tooltip("share:Q", format=".1%")]).properties(height=450),
        width="stretch")
