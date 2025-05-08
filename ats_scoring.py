import argparse
import json
import logging
import warnings
import os
import re
from pathlib import Path
import docx
from typing import List, Dict
warnings.filterwarnings("ignore")  # I hide warnings for clarity
for lib in ["accelerate", "transformers", "sentence_transformers"]:
    logging.getLogger(lib).setLevel(logging.ERROR)
try:
    from sentence_transformers import SentenceTransformer, util
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _sbert = None  # You disable semantic similarity if model load fails
_WORD = re.compile(r"\b\w+\b", re.UNICODE)

def _extract_text(path: Path) -> str:
    # She reads all paragraphs from the DOCX
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def _keyword_overlap(resume: str, jd: str) -> float:
    # I compute Jaccard on word sets
    a = set(_WORD.findall(resume.lower()))
    b = set(_WORD.findall(jd.lower()))
    return len(a & b) / len(a | b) if a and b else 0.0

def _semantic_similarity(resume: str, jd: str) -> float:
    # You use SBERT embeddings if available
    if _sbert is None:
        return 0.0
    emb = _sbert.encode([resume, jd], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())

def _score(path: Path, jd: str) -> Dict[str, float]:
    # I score each resume against its JD
    text = _extract_text(path)
    kw = _keyword_overlap(text, jd)
    sem = _semantic_similarity(text, jd)
    return {"keyword_overlap": kw, "semantic_similarity": sem, "ats_score": (kw + sem) / 2}

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default="pipeline/ats_results")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = json.loads(Path(args.manifest).read_text("utf-8"))
    results = []
    for entry in manifest:
        if not entry.get("compiled"):
            continue  # He skips failed resume compilations
        docx_path = Path(entry["docx"])
        scores = _score(docx_path, entry["job_description"])
        scores.update({"profile_index": entry["profile_index"], "docx": str(docx_path)})
        logging.info("%s %.3f", docx_path.name, scores["ats_score"])
        results.append(scores)

    (out_dir / "ats_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
