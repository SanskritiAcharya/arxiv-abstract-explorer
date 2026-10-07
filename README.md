# arXiv Abstract Explorer: DCS 401 Project

NLP on 287,421 arXiv paper abstracts ([Kaggle dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset)).
It asks how strongly an abstract's text is tied to its research category, predicts the
category of any abstract, and highlights the words behind each prediction.

## Architecture

```
arXiv CSV (410 MB)
   │  src/preprocess.py   read in 25k-row chunks → clean in parallel → data/chunks/*.parquet
   ▼                      de-duplicate, keep 13 categories with ≥ 2,000 papers
processed.parquet (260,169 abstracts)
   ├─ src/analyze.py      chi-square word↔category, category similarity, length (η²)
   └─ src/train.py        TF-IDF (words + 2-word phrases) → Naive Bayes baseline + Logistic Regression
          │
          ▼
api/main.py (FastAPI)  ── /predict, /sample, /stats/* ──►  app/streamlit_app.py (Streamlit UI)
```

## Results (20% held-out test set, 52,034 abstracts)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Complement Naive Bayes (baseline) | 0.737 | 0.463 |
| Logistic Regression (class-balanced) | 0.735 | 0.634 |

Key finding: the categories whose abstracts use the most similar language are the ones the model
confuses most (cs.LG ↔ stat.ML similarity 0.90, cs.CV ↔ eess.IV 0.84). Abstract *length*
barely depends on category (η² = 0.027). The link is in *which words* are used.

## Setup

```bash
uv venv -p 3.13 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['wordnet','omw-1.4','stopwords']]"
```

Download the dataset into `data/` (no Kaggle login needed):

```bash
curl -L -o data/arxiv.zip https://www.kaggle.com/api/v1/datasets/download/sumitm004/arxiv-scientific-research-papers-dataset
unzip data/arxiv.zip -d data && rm data/arxiv.zip
```

## Run

```bash
# 1. build the data and model (~5 minutes total; only needed once)
.venv/bin/python -m src.preprocess
.venv/bin/python -m src.analyze
.venv/bin/python -m src.train

# 2. start the API (terminal 1); interactive docs at http://127.0.0.1:8000/docs
.venv/bin/uvicorn api.main:app

# 3. start the dashboard (terminal 2); opens http://localhost:8501
.venv/bin/streamlit run app/streamlit_app.py
```

If `API_URL` is not set, the Streamlit app starts the FastAPI backend itself in a background
thread, so `streamlit run app/streamlit_app.py` alone is also enough.

## Deployment (Streamlit Community Cloud)
The trained model (`models/`) and analysis results (`artifacts/`) are committed, so the cloud app
needs no dataset or training. Deploy with main file `app/streamlit_app.py` and Python 3.13.

## Known limitations
- Mostly computer-science papers; 134 small categories were excluded.
- Each paper has one label, but many are really cross-disciplinary, so some "errors" are arguable.
- 2025–2026 papers dominate the data (about 53% of rows), so the model reflects recent vocabulary (LLMs, agents).
