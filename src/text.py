"""Text cleaning shared by preprocessing, training and the API.

Keeping this in one place guarantees the model sees text cleaned exactly the
same way at training time and at prediction time.
"""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:  # first run on a fresh machine or cloud host
        nltk.download(_pkg, quiet=True)

LATEX_RE = re.compile(r"\$[^$]*\$")          # inline maths like $O(n^2)$
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")  # words of 2+ characters

_lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english")) | {
    # words that appear in almost every abstract and carry no topic signal
    "paper", "propose", "proposed", "show", "result", "method", "approach",
    "also", "use", "using", "used", "based", "new", "work", "study", "present",
}


@lru_cache(maxsize=500_000)
def lemma(word: str) -> str:
    return _lemmatizer.lemmatize(word.lower())


def tokens_with_spans(text: str):
    """Yield (start, end, lemma, keep) for every word in the raw text.

    `keep` is False for stop words and words inside LaTeX maths. The spans let
    the API map model weights back onto the original abstract for highlighting.
    """
    masked = LATEX_RE.sub(lambda m: " " * len(m.group()), text)
    for m in TOKEN_RE.finditer(masked):
        lem = lemma(m.group())
        keep = lem not in STOPWORDS and m.group().lower() not in STOPWORDS
        yield m.start(), m.end(), lem, keep


def clean(text: str) -> str:
    """Lower-case, strip LaTeX, tokenize, drop stop words and lemmatize."""
    return " ".join(lem for _, _, lem, keep in tokens_with_spans(text) if keep)
