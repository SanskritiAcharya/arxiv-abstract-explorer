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


def api_healthy(url):
    try:
        return requests.get(f"{url}/health", timeout=2).ok
    except requests.RequestException:
        return False


@st.cache_resource
def start_embedded_api():
    """On hosts that run only Streamlit (e.g. Streamlit Community Cloud), start the
    FastAPI backend in a background thread inside the same container."""
    url = "http://127.0.0.1:8000"
    if api_healthy(url):  # already running, e.g. left over from before a hot reload
        return url
    import uvicorn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(240):
        if api_healthy(url):
            return url
        time.sleep(0.5)
    raise RuntimeError("Embedded API did not start")  # not cached, so the next run retries


# Set API_URL to use a separately running backend; otherwise one is started here.
try:
    API = os.environ.get("API_URL") or start_embedded_api()
except RuntimeError:
    st.error("The server is still starting. Please refresh in a few seconds.")
    st.stop()


@st.cache_data(ttl=600)
def get(path):
    for attempt in range(5):
        try:
            r = requests.get(f"{API}{path}", timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            if attempt == 4:
                raise
            time.sleep(2)


BAR, SOFT = "#1F2937", "#CBD5E1"
TOWARDS, AWAY = "22, 163, 74", "220, 38, 38"

st.html("""
<style>
  .block-container { padding-top: 3rem; max-width: 1100px; }
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="stDecoration"], footer { display: none; }
  h1, h2, h3, h4 { letter-spacing: -0.01em; }
  .topbar { display: flex; align-items: baseline; justify-content: space-between;
            padding-bottom: 0.9rem; border-bottom: 1px solid #E5E7EB; margin-bottom: 0.2rem; }
  .brand { font-size: 1.25rem; font-weight: 700; color: #111827; }
  .brand span { color: #6B7280; font-weight: 500; }
  .sub { color: #6B7280; font-size: 0.88rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #E5E7EB; }
  .stTabs [data-baseweb="tab"] { padding: 0.7rem 0; font-weight: 500; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 0.4rem 0 0.6rem; }
  .kpi { border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 16px; }
  .kpi .v { font-size: 1.45rem; font-weight: 600; color: #111827; }
  .kpi .l { font-size: 0.8rem; color: #6B7280; }
  .label { font-size: 0.8rem; color: #6B7280; font-weight: 500; margin-bottom: 4px; }
  .pred { font-size: 1.7rem; font-weight: 600; color: #111827; line-height: 1.2; }
  .pred-name { color: #4B5563; margin: 2px 0 14px; }
  .pill { display: inline-block; font-size: 0.72rem; font-weight: 500; padding: 2px 9px;
          border-radius: 999px; margin-left: 8px; vertical-align: middle; }
  .pill.ok { background: #DCFCE7; color: #166534; }
  .pill.no { background: #FEE2E2; color: #991B1B; }
  .bar { display: grid; grid-template-columns: 70px 1fr 40px; align-items: center; gap: 10px;
         font-size: 0.85rem; margin: 7px 0; color: #374151; }
  .track { background: #F1F5F9; height: 6px; border-radius: 3px; }
  .fill { background: #111827; height: 6px; border-radius: 3px; }
  .bar.rest .fill { background: #CBD5E1; }
  .abstract { line-height: 1.85; font-size: 0.97rem; color: #1F2937; }
  .key { font-size: 0.78rem; color: #6B7280; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 12px; }
  .paper { font-weight: 600; color: #111827; margin-top: 2px; }
  .muted { color: #6B7280; font-size: 0.85rem; }
</style>
""")


def card(title):
    c = st.container(border=True)
    c.markdown(f"**{title}**")
    return c


def bar_chart(df, x, y, title, color=BAR, domain=None, tooltip=None):
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    return alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=2, height={"band": 0.7}).encode(
        x=alt.X(x, title=title, scale=scale), y=alt.Y(y, sort="-x", title=None),
        tooltip=tooltip or [y, x]).configure_axis(grid=False, domain=False, ticks=False, labelColor="#4B5563",
                                                  titleColor="#6B7280", titleFontWeight="normal")


def highlight_html(segments):
    """Green = pushed towards the predicted category, red = pushed away."""
    peak = max((abs(s["score"]) for s in segments), default=0) or 1
    parts = []
    for seg in segments:
        text = html.escape(seg["text"])
        strength = abs(seg["score"]) / peak
        if strength < 0.2:
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
    st.error("The server is still starting. Please refresh in a few seconds.")
    st.stop()

label = lambda code: f"{code} · {names.get(code, code)}"
ov, assoc, mt = get("/stats/overview"), get("/stats/association"), get("/stats/model")
pre, best_model = ov["preprocess"], mt["results"][-1]


def kpis(items):
    st.html('<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                                          for v, l in items) + "</div>")


st.html('<div class="topbar"><div class="brand">arXiv Abstract Explorer</div>'
        '<div class="sub">Research field classification from paper abstracts</div></div>')

tab_classify, tab_explore, tab_model = st.tabs(["Classify", "Analysis", "Model"])

def load_example():
    pick = st.session_state.pick
    s = requests.get(f"{API}/sample" + ("" if pick == "any" else f"?category={pick}"), timeout=30).json()
    st.session_state.abstract = s["summary"]
    st.session_state.truth = (s["title"], s["category_code"])


# ---------------------------------------------------------------- classify
with tab_classify:
    if "abstract" not in st.session_state:
        st.session_state.abstract, st.session_state.truth = "", None

    text = st.text_area("Abstract", key="abstract", height=200, label_visibility="collapsed",
                        placeholder="Paste a paper abstract…")
    if st.session_state.truth:
        title, code = st.session_state.truth
        st.html(f'<div class="paper">{html.escape(title)}</div>'
                f'<div class="muted">arXiv category: {label(code)}</div>')

    b1, b2, b3 = st.columns([1.2, 2.2, 1.4], vertical_alignment="bottom")
    run = b1.button("Classify", type="primary", width="stretch", disabled=len(text.strip()) < 30)
    b2.selectbox("Example category", ["any", *sorted(names)], key="pick", label_visibility="collapsed",
                 format_func=lambda c: "Example from any category" if c == "any" else label(c))
    b3.button("Load example", width="stretch", on_click=load_example)

    if run:
        r = requests.post(f"{API}/predict", json={"text": text, "top_k": 3}, timeout=30)
        if r.status_code != 200:
            st.warning(r.json().get("detail", "Prediction failed."))
        else:
            res = r.json()
            best = res["predictions"][0]
            c1, c2 = st.columns([1, 2])
            with c1, st.container(border=True, height="stretch"):
                pill = ""
                if st.session_state.truth:
                    ok = best["code"] == st.session_state.truth[1]
                    pill = f'<span class="pill {"ok" if ok else "no"}">{"Correct" if ok else "Incorrect"}</span>'
                bars = "".join(
                    f'<div class="bar{"" if i == 0 else " rest"}"><span>{p["code"]}</span>'
                    f'<div class="track"><div class="fill" style="width:{p["probability"]:.0%}"></div></div>'
                    f'<span>{p["probability"]:.0%}</span></div>' for i, p in enumerate(res["predictions"]))
                st.html(f'<div class="label">Prediction</div><div class="pred">{best["code"]}{pill}</div>'
                        f'<div class="pred-name">{html.escape(best["name"])}</div>{bars}')
            with c2, st.container(border=True, height="stretch"):
                st.html(f'<div class="label">Word influence'
                        f'<span class="dot" style="background: rgb({TOWARDS})"></span>supports'
                        f'<span class="dot" style="background: rgb({AWAY})"></span>opposes</div>'
                        + highlight_html(res["segments"]))

# ---------------------------------------------------------------- explore
with tab_explore:
    kpis([(f"{pre['raw_rows']:,}", "Raw papers"), (f"{pre['rows_after_cleaning']:,}", "After cleaning"),
          (str(pre["categories_kept"]), "Categories"), (f"{assoc['length']['eta_squared']:.3f}", "Length η²")])

    counts = pd.DataFrame(ov["category_counts"].items(), columns=["code", "papers"])
    counts["name"] = counts["code"].map(names)
    left, right = st.columns(2)
    with left, card("Papers per category"):
        st.altair_chart(bar_chart(counts, "papers:Q", "code:N", None,
                                  tooltip=["name", alt.Tooltip("papers:Q", format=",")]), width="stretch")
    with right, card("Average abstract length"):
        length = pd.DataFrame(assoc["length"]["by_category"])
        st.altair_chart(bar_chart(length, "mean:Q", "category_code:N", None, color=SOFT,
                                  tooltip=["category_code", alt.Tooltip("mean:Q", format=".0f")]),
                        width="stretch")

    with card("Top terms by category"):
        code = st.selectbox("Category", sorted(assoc["top_terms"]), format_func=label,
                            label_visibility="collapsed")
        terms = pd.DataFrame(assoc["top_terms"][code])
        st.altair_chart(bar_chart(terms, "chi2:Q", "word:N", "chi-square",
                                  tooltip=["word", alt.Tooltip("chi2:Q", format=",.0f")]), width="stretch")

    with card("Vocabulary similarity"):
        sim = assoc["similarity"]
        sim_df = pd.DataFrame(sim["matrix"], index=sim["codes"], columns=sim["codes"]).stack().reset_index()
        sim_df.columns = ["a", "b", "similarity"]
        st.altair_chart(alt.Chart(sim_df).mark_rect(cornerRadius=2).encode(
            x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None),
            color=alt.Color("similarity:Q", title=None, scale=alt.Scale(range=["#F8FAFC", "#111827"])),
            tooltip=["a", "b", alt.Tooltip("similarity:Q", format=".2f")]).properties(height=420)
            .configure_axis(domain=False, ticks=False, labelColor="#4B5563"), width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    kpis([(f"{best_model['accuracy']:.1%}", "Accuracy"), (f"{best_model['macro_f1']:.3f}", "Macro F1"),
          (f"{mt['train_size']:,}", "Training abstracts"), (f"{mt['test_size']:,}", "Test abstracts")])

    summary = pd.DataFrame([{k: r[k] for k in ("model", "accuracy", "macro_f1", "weighted_f1")}
                            for r in mt["results"]])
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "model": "Model",
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
        "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
        "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f")})

    per = pd.DataFrame([{"code": c, **best_model["per_class"][c]} for c in mt["labels"]])
    left, right = st.columns([2, 3])
    with left, card("F1 by category"):
        st.altair_chart(bar_chart(per, "f1-score:Q", "code:N", None, domain=[0, 1], tooltip=[
            "code", alt.Tooltip("precision:Q", format=".2f"), alt.Tooltip("recall:Q", format=".2f"),
            alt.Tooltip("f1-score:Q", format=".2f")]).properties(height=420), width="stretch")
    with right, card("Confusion matrix"):
        cm = pd.DataFrame(best_model["confusion_matrix"], index=mt["labels"], columns=mt["labels"])
        cm = cm.div(cm.sum(axis=1), axis=0).stack().reset_index()
        cm.columns = ["actual", "predicted", "share"]
        st.altair_chart(alt.Chart(cm).mark_rect(cornerRadius=2).encode(
            x=alt.X("predicted:N", title="Predicted"), y=alt.Y("actual:N", title="Actual"),
            color=alt.Color("share:Q", title=None, scale=alt.Scale(range=["#F8FAFC", "#111827"])),
            tooltip=["actual", "predicted", alt.Tooltip("share:Q", format=".1%")]).properties(height=420)
            .configure_axis(domain=False, ticks=False, labelColor="#4B5563", titleColor="#6B7280",
                            titleFontWeight="normal"), width="stretch")
