#!/usr/bin/env python3
"""
ats_scoring.py – Real-world-style ATS Resume Scorer (0–100)
===========================================================

Features
--------
1. **Structured parsing (spaCy NER)** – detects name, e-mail, phone in the CV.
2. **Context-aware keyword weighting** – simple overlap metric today.
3. **Semantic similarity** – SBERT (`all-MiniLM-L6-v2`) by default, `--tfidf`
   flag for a no-GPU fallback.
4. **ATS-compliance checks** – flags tables, images, or missing contact info;
   if anything is amiss, the final score is *halved*.
5. **CUDA → CPU fix** – converts PyTorch tensors to NumPy before calling
   `sklearn.metrics.pairwise.cosine_similarity`, avoiding the
   “can't convert cuda tensor to numpy” crash.

Score formula::

    base = 0.70 * semantic_similarity + 0.30 * keyword_coverage
    final = base * 100   (halve if compliance issues)

Usage
-----
    python ats_scoring.py -j JD.pdf -r CV.docx          # SBERT mode (GPU/CPU)
    python ats_scoring.py -j jd.docx -r cv.pdf --tfidf  # classic TF-IDF

Extra deps: `python-docx`, `pdfplumber`, `spacy`, `sentence-transformers`
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ───────────────────────── TEXT EXTRACTION ──────────────────────────


def _extract_text(path: Path) -> Tuple[str, Dict[str, bool]]:
    """Return (plain_text, format_info)."""
    info = {"has_tables": False, "has_images": False}
    suffix = path.suffix.lower()

    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:
            raise ImportError(
                "python-docx not installed. `pip install python-docx`"
            ) from e
        doc = docx.Document(path)
        if doc.tables:
            info["has_tables"] = True
        text = "\n".join(p.text for p in doc.paragraphs)

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber not installed. `pip install pdfplumber`"
            ) from e
        pages_text: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page.images:
                    info["has_images"] = True
                if page.extract_tables():
                    info["has_tables"] = True
                pages_text.append(page.extract_text() or "")
        text = "\n".join(pages_text)

    else:
        raise ValueError(
            f"Unsupported file type: {suffix} – use .docx or .pdf")

    return text, info


# ───────────────────────── NORMALISATION ──────────────────────────

_STOPS: set[str] | None = None


def _normalise(txt: str) -> str:
    """Lower-case, strip punctuation, remove stop-words."""
    global _STOPS
    if _STOPS is None:
        import nltk

        try:
            _STOPS = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            _STOPS = set(nltk.corpus.stopwords.words("english"))

    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [t for t in txt.split() if t not in _STOPS and len(t) > 2]
    return " ".join(tokens)


# ───────────────────────── STRUCTURED PARSE ────────────────────────


def _structured_parse(raw_text: str) -> Dict[str, bool]:
    """Detect presence of PERSON / email / phone – auto-downloads model if needed."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy not installed. `pip install spacy` and "
            "`python -m spacy download en_core_web_sm`"
        ) from e

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli

        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(raw_text)

    flags = {
        "has_email": bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)),
        "has_phone": bool(re.search(r"(\+?\d[\d\-\s\(\)]{7,}\d)", raw_text)),
        "has_name": any(ent.label_ == "PERSON" for ent in doc.ents),
    }
    return flags


# ───────────────────────── VECTORISATION ──────────────────────────


def _vectorise(texts: List[str], use_tfidf: bool):
    """Return (vectors, model_or_None)."""
    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer()
        return vec.fit_transform(texts), None
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. `pip install sentence-transformers`"
            ) from e

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = model.encode(texts, convert_to_tensor=True,
                            normalize_embeddings=True)
        return vecs, model


# ───────────────────────── KEYWORD COVERAGE ────────────────────────


def _keyword_coverage(jd_text: str, cv_text: str) -> float:
    jd_tokens = set(jd_text.split())
    cv_tokens = set(cv_text.split())
    if not jd_tokens:
        return 0.0
    return len(jd_tokens & cv_tokens) / len(jd_tokens)


# ───────────────────────── COSINE SIMILARITY ───────────────────────


def _cosine(a, b) -> float:
    """
    Cosine similarity wrapper that handles:
      • SciPy sparse matrices (TF-IDF path)
      • PyTorch tensors (SBERT path – CPU or CUDA)
      • NumPy arrays
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import torch

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if hasattr(x, "toarray"):           # sparse matrix
            return x.toarray()
        return np.asarray(x)

    a_np, b_np = to_numpy(a), to_numpy(b)
    return float(cosine_similarity(a_np, b_np)[0, 0])


# ───────────────────────── CLI & MAIN ──────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="ATS-style résumé scorer (0–100)")
    p.add_argument(
        "--jd", "-j", required=True, type=Path, help="Job description (.docx|.pdf)"
    )
    p.add_argument(
        "--resume", "-r", required=True, type=Path, help="Resume (.docx|.pdf)"
    )
    p.add_argument("--tfidf", action="store_true",
                   help="Use TF-IDF instead of SBERT")
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.jd.exists() or not args.resume.exists():
        sys.exit("One or both input files do not exist.")

    jd_raw, _ = _extract_text(args.jd)
    cv_raw, fmt_info = _extract_text(args.resume)

    # compliance: missing contact info / tables / images
    flags = _structured_parse(cv_raw)

    jd_clean = _normalise(jd_raw)
    cv_clean = _normalise(cv_raw)

    vecs, _ = _vectorise([jd_clean, cv_clean], use_tfidf=args.tfidf)

    similarity = _cosine(vecs[0:1], vecs[1:2])        # 0–1
    coverage = _keyword_coverage(jd_clean, cv_clean)  # 0–1

    base_score = 0.70 * similarity + 0.30 * coverage
    score_0_100 = int(round(base_score * 100))

    compliance_ok = all(flags.values()) and not (
        fmt_info["has_tables"] or fmt_info["has_images"]
    )
    if not compliance_ok:
        score_0_100 //= 2  # halve

    # ── Pretty output ───────────────────────────────────────────────
    print(f"Semantic similarity : {similarity:.3f}")
    print(f"Keyword coverage     : {coverage:.3f}")

    if not compliance_ok:
        missing = [k for k, v in flags.items() if not v]
        fmt_issues: list[str] = []
        if fmt_info["has_tables"]:
            fmt_issues.append("tables")
        if fmt_info["has_images"]:
            fmt_issues.append("images")
        issues = ", ".join(missing + fmt_issues)
        print(f"Compliance issues   : {issues} → score halved")
    else:
        print(f"Compliance None → score kept")

    print(f"\nATS-style SCORE     : {score_0_100}/100")


if __name__ == "__main__":
    main()
