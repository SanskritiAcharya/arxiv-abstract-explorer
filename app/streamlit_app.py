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

st.set_page_config(page_title="arXiv Abstract Explorer", layout="wide")


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


ACCENT, MUTED, LIGHT = "#B31B1B", "#5B6770", "#A8A29A"
TOWARDS, AWAY = "47, 111, 74", "179, 27, 27"

st.html("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  .masthead { border-bottom: 1px solid #DDD8CE; padding-bottom: 1rem; margin-bottom: 0.4rem; }
  .masthead .eyebrow { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #5B6770; }
  .masthead h1 { font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 2.3rem;
                 margin: 0.15rem 0 0.35rem 0; padding: 0; color: #1F1F1F; }
  .masthead p { color: #4A4A4A; margin: 0; max-width: 760px; }
  .facts { display: flex; gap: 2.2rem; margin: 0.9rem 0 0 0; flex-wrap: wrap; }
  .facts div { font-size: 0.82rem; color: #5B6770; }
  .facts b { display: block; font-size: 1.15rem; font-weight: 600; color: #1F1F1F; }
  .fig { font-size: 0.85rem; color: #5B6770; margin-top: -0.4rem; }
  .fig b { color: #1F1F1F; }
  .pred-code { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 500; color: #B31B1B; }
  .pred-name { color: #4A4A4A; margin-bottom: 0.6rem; }
  .badge { display: inline-block; font-size: 0.75rem; padding: 1px 8px; border-radius: 3px;
           border: 1px solid; margin-left: 6px; vertical-align: middle; }
  .badge.ok { color: #2F6F4A; border-color: #2F6F4A; }
  .badge.no { color: #B31B1B; border-color: #B31B1B; }
  .prob-row { display: grid; grid-template-columns: 78px 1fr 44px; align-items: center; gap: 8px;
              font-size: 0.85rem; margin: 4px 0; }
  .prob-row code { background: none; padding: 0; }
  .prob-track { background: #ECE8E0; height: 8px; border-radius: 2px; }
  .prob-fill { background: #B31B1B; height: 8px; border-radius: 2px; }
  .prob-row.rest .prob-fill { background: #A8A29A; }
  .abstract { line-height: 1.95; font-size: 1rem; font-family: 'Source Serif 4', serif; }
  .legend { font-size: 0.8rem; color: #5B6770; margin-bottom: 0.5rem; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px;
            vertical-align: -1px; }
  .terms code { margin-right: 4px; }
  .paper-title { font-family: 'Source Serif 4', serif; font-size: 1.02rem; margin: 0.2rem 0 0.1rem; }
  footer, [data-testid="stDecoration"] { display: none; }
</style>
""")


def figure(n, title, note):
    st.html(f'<div class="fig"><b>Figure {n}.</b> {title} {note}</div>')


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
        colour = TOWARDS if seg["score"] > 0 else AWAY
        alpha = 0.12 + 0.55 * strength
        parts.append(f'<span title="{seg["score"]:+.3f}" style="background: rgba({colour}, {alpha:.2f});'
                     f' border-radius: 3px; padding: 0 2px;">{text}</span>')
    return f'<div class="abstract">{"".join(parts)}</div>'


try:
    names = get("/categories")
except requests.RequestException:
    st.error(f"Can't reach the API at {API}. Start it with `uvicorn api.main:app` and refresh.")
    st.stop()

label = lambda code: f"{code} · {names.get(code, code)}"
ov, assoc, mt = get("/stats/overview"), get("/stats/association"), get("/stats/model")
pre, best_model = ov["preprocess"], mt["results"][-1]

st.html(f"""
<div class="masthead">
  <div class="eyebrow">DCS 401 · Natural Language Processing · Course project</div>
  <h1>arXiv Abstract Explorer</h1>
  <p>How strongly is the text of a paper's abstract tied to its research category? This tool classifies
     abstracts, shows which words drive each prediction, and summarises the statistics behind it.</p>
  <div class="facts">
    <div><b>{pre['rows_after_cleaning']:,}</b>abstracts after cleaning</div>
    <div><b>{pre['categories_kept']}</b>arXiv categories</div>
    <div><b>{mt['n_features']:,}</b>TF-IDF features</div>
    <div><b>{best_model['accuracy']:.1%}</b>test accuracy</div>
    <div><b>{best_model['macro_f1']:.3f}</b>macro F1</div>
  </div>
</div>
""")

with st.sidebar:
    st.markdown("### About")
    st.markdown("Built on the Kaggle arXiv abstracts dataset. Abstracts are cleaned (lower-casing, "
                "stop-word removal, lemmatisation), vectorised with TF-IDF over words and two-word "
                "phrases, and classified with a class-weighted Logistic Regression.")
    st.markdown("### Method")
    st.markdown("- Chi-square test for category-specific words\n"
                "- Cosine similarity between category centroids\n"
                "- η² for abstract length vs. category\n"
                "- Complement Naive Bayes as baseline")
    st.markdown("### Stack")
    st.markdown("Python · scikit-learn · NLTK · FastAPI · Streamlit")
    st.caption("Highlights are per-word contributions to the predicted class "
               "(TF-IDF weight × model coefficient).")

tab_classify, tab_explore, tab_model = st.tabs(["Classify", "Dataset analysis", "Model evaluation"])

# ---------------------------------------------------------------- classify
with tab_classify:
    if "abstract" not in st.session_state:
        st.session_state.abstract, st.session_state.truth = "", None

    left, right = st.columns([3, 1], gap="large")
    with right:
        st.markdown("**Load an example**")
        st.caption("Papers from the held-out test set, which the model never saw during training.")
        pick = st.selectbox("Category", ["any category", *sorted(names)], label_visibility="collapsed",
                            format_func=lambda c: "Any category" if c == "any category" else label(c))
        if st.button("Load random paper", width="stretch"):
            params = "" if pick == "any category" else f"?category={pick}"
            s = requests.get(f"{API}/sample{params}", timeout=30).json()
            st.session_state.abstract = s["summary"]
            st.session_state.truth = (s["title"], s["category_code"])
    with left:
        text = st.text_area("Abstract", key="abstract", height=220,
                            placeholder="Paste the abstract of a research paper (at least 30 characters).")
        if st.session_state.truth:
            title, code = st.session_state.truth
            st.html(f'<div class="paper-title">{html.escape(title)}</div>'
                    f'<div class="legend">Listed on arXiv under <code>{code}</code> · '
                    f'{html.escape(names.get(code, code))}</div>')
        run = st.button("Classify abstract", type="primary", disabled=len(text.strip()) < 30)

    if run:
        r = requests.post(f"{API}/predict", json={"text": text, "top_k": 3}, timeout=30)
        if r.status_code != 200:
            st.warning(r.json().get("detail", "Prediction failed."))
        else:
            res = r.json()
            best = res["predictions"][0]
            st.divider()
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                badge = ""
                if st.session_state.truth:
                    ok = best["code"] == st.session_state.truth[1]
                    badge = ('<span class="badge ok">matches arXiv label</span>' if ok
                             else '<span class="badge no">differs from arXiv label</span>')
                rows = "".join(
                    f'<div class="prob-row{"" if i == 0 else " rest"}"><code>{p["code"]}</code>'
                    f'<div class="prob-track"><div class="prob-fill" style="width:{p["probability"]:.0%}">'
                    f'</div></div><span>{p["probability"]:.0%}</span></div>'
                    for i, p in enumerate(res["predictions"]))
                st.html(f'<div class="legend">Predicted category</div>'
                        f'<div class="pred-code">{best["code"]}{badge}</div>'
                        f'<div class="pred-name">{html.escape(best["name"])}</div>'
                        f'<div class="legend">Top {len(res["predictions"])} probabilities</div>{rows}')
            with c2:
                st.html(f'<div class="legend">Word contributions towards <code>{best["code"]}</code>'
                        f'<span class="swatch" style="background: rgba({TOWARDS}, .6)"></span>supports'
                        f'<span class="swatch" style="background: rgba({AWAY}, .6)"></span>opposes'
                        f' · hover a word for its score</div>')
                with st.container(border=True):
                    st.html(highlight_html(res["segments"]))
                if res["top_words"]:
                    st.html('<div class="legend terms">Strongest evidence: ' +
                            "".join(f"<code>{html.escape(w['word'])}</code>" for w in res["top_words"]) +
                            "</div>")

# ---------------------------------------------------------------- explore
with tab_explore:
    st.markdown("#### Data")
    st.markdown(f"{pre['raw_rows']:,} raw papers were reduced to **{pre['rows_after_cleaning']:,}** after "
                f"removing duplicates and empty abstracts, keeping **{pre['categories_kept']}** of "
                f"{pre['categories_total']} categories. The CSV was read in 25,000-row chunks and cleaned "
                f"in parallel ({pre['seconds']:.0f}s).")

    counts = pd.DataFrame(ov["category_counts"].items(), columns=["code", "papers"])
    counts["name"] = counts["code"].map(names)
    st.altair_chart(alt.Chart(counts).mark_bar(color=MUTED).encode(
        x=alt.X("papers:Q", title="papers"), y=alt.Y("code:N", sort="-x", title=None),
        tooltip=["name", alt.Tooltip("papers:Q", format=",")]), width="stretch")
    figure(1, "Papers per category.", "The data is heavily imbalanced: four AI fields make up most of it, "
           "so the classifier uses class weighting.")

    st.markdown("#### Category-specific vocabulary")
    code = st.selectbox("Category", sorted(assoc["top_terms"]), format_func=label)
    terms = pd.DataFrame(assoc["top_terms"][code])
    st.altair_chart(alt.Chart(terms).mark_bar(color=ACCENT).encode(
        x=alt.X("chi2:Q", title="chi-square score"),
        y=alt.Y("word:N", sort="-x", title=None),
        tooltip=["word", alt.Tooltip("chi2:Q", format=",.0f"),
                 alt.Tooltip("pct_in_category", title="% of papers in category"),
                 alt.Tooltip("pct_elsewhere", title="% of papers elsewhere")]), width="stretch")
    figure(2, f"Terms most associated with {code}.", "Higher chi-square means the term's presence depends "
           "more strongly on the category.")

    st.markdown("#### Similarity between categories")
    sim = assoc["similarity"]
    sim_df = pd.DataFrame(sim["matrix"], index=sim["codes"], columns=sim["codes"]).stack().reset_index()
    sim_df.columns = ["a", "b", "similarity"]
    st.altair_chart(alt.Chart(sim_df).mark_rect().encode(
        x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None),
        color=alt.Color("similarity:Q", scale=alt.Scale(range=["#F7F4EE", "#5B6770", "#1F2A33"])),
        tooltip=["a", "b", alt.Tooltip("similarity:Q", format=".2f")]).properties(height=450),
        width="stretch")
    figure(3, "Cosine similarity of each category's mean TF-IDF vector.", "Dark off-diagonal cells "
           "(e.g. cs.LG and stat.ML) are fields that write alike; these are the pairs the model confuses most.")

    st.markdown("#### Abstract length")
    length = pd.DataFrame(assoc["length"]["by_category"])
    st.altair_chart(alt.Chart(length).mark_bar(color=MUTED).encode(
        x=alt.X("mean:Q", title="mean words per abstract"),
        y=alt.Y("category_code:N", sort="-x", title=None),
        tooltip=["category_code", alt.Tooltip("mean:Q", format=".0f"), "median"]), width="stretch")
    eta = assoc["length"]["eta_squared"]
    figure(4, "Mean abstract length by category.", f"η² = {eta:.3f}: category explains only {eta:.1%} of "
           "the variation in length. The signal is in which words are used, not how many.")

# ---------------------------------------------------------------- model
with tab_model:
    st.markdown(f"Trained on **{mt['train_size']:,}** abstracts and evaluated on **{mt['test_size']:,}** "
                f"held-out abstracts using {mt['n_features']:,} TF-IDF features (unigrams and bigrams).")
    summary = pd.DataFrame([{k: r[k] for k in ("model", "accuracy", "macro_f1", "weighted_f1")}
                            for r in mt["results"]])
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "model": "Model",
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
        "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
        "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f")})
    st.caption("Table 1. Macro F1 weights every category equally, so it exposes weakness on small "
               "categories that accuracy hides.")

    per = pd.DataFrame([{"code": c, **best_model["per_class"][c]} for c in mt["labels"]])
    st.markdown("#### F1 score per category")
    st.altair_chart(alt.Chart(per).mark_bar(color=ACCENT).encode(
        x=alt.X("f1-score:Q", title="F1 score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("code:N", sort="-x", title=None),
        tooltip=["code", alt.Tooltip("precision:Q", format=".2f"), alt.Tooltip("recall:Q", format=".2f"),
                 alt.Tooltip("f1-score:Q", format=".2f"), alt.Tooltip("support:Q", format=",")]),
        width="stretch")
    figure(5, "Per-category F1 for Logistic Regression.", "Categories that share vocabulary with a larger "
           "one (eess.IV with cs.CV, stat.ML with cs.LG) are among the weakest.")

    st.markdown("#### Confusion matrix")
    cm = pd.DataFrame(best_model["confusion_matrix"], index=mt["labels"], columns=mt["labels"])
    cm = cm.div(cm.sum(axis=1), axis=0).stack().reset_index()
    cm.columns = ["actual", "predicted", "share"]
    st.altair_chart(alt.Chart(cm).mark_rect().encode(
        x=alt.X("predicted:N", title="predicted"), y=alt.Y("actual:N", title="actual"),
        color=alt.Color("share:Q", title="share", scale=alt.Scale(range=["#F7F4EE", "#D98A7E", "#8E1515"])),
        tooltip=["actual", "predicted", alt.Tooltip("share:Q", format=".1%")]).properties(height=450),
        width="stretch")
    figure(6, "Row-normalised confusion matrix.", "Each row shows where papers of one actual category "
           "were assigned.")
