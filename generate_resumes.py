import argparse
import json
import random
import uuid
import logging
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")  # I suppress unrelated warnings for cleaner logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # You disable parallel tokenizer threads
for lib in ["accelerate", "transformers", "sentence_transformers"]:
    logging.getLogger(lib).setLevel(logging.ERROR)  # He silences verbose library logs
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging as _hf
_hf.set_verbosity_error()  # Third person: The transformer logs are set to error-only
try:
    import pypandoc  # I try to import Pandoc for DOCX conversion
except ImportError:
    pypandoc = None  # If missing, conversion will be skipped
_tok = None
_model = None
_pipe = None

def _get_pipe():
    # I lazily initialize the Llama-2 text-generation pipeline
    global _tok, _model, _pipe
    if _pipe is None:
        name = "meta-llama/Llama-2-13b-chat-hf"
        # You fetch tokenizer and model with trust_remote_code for custom layers
        _tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", trust_remote_code=True)
        # He sets deterministic generation with sampling enabled
        _pipe = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tok,
            max_new_tokens=700,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
    return _pipe

def generate_resume_latex(profile, job_desc):
    # The function calls the LLM to produce LaTeX content
    pipe = _get_pipe()
    prompt = (
        "You are an expert resume writer. Given the applicant profile JSON and the plain text job description, "
        "create a one-page resume in LaTeX using the article class. Output only LaTeX without markdown."
        f"\nAPPLICANT PROFILE:\n{json.dumps(profile)}\nJOB DESCRIPTION:\n{job_desc}\nLaTeX:"
    )
    output = pipe(prompt)[0]["generated_text"]
    if "LaTeX:" in output:
        # You strip the echoed prompt to isolate pure LaTeX
        output = output.split("LaTeX:")[-1]
    return output.strip()

def compile_latex_to_docx(latex, tex_path, docx_path):
    # She writes the LaTeX to a file before conversion
    tex_path.write_text(latex, encoding="utf-8")
    if pypandoc is None:
        return False  # Pandoc missing means failure
    try:
        # I invoke Pandoc quietly to convert .tex to .docx
        pypandoc.convert_file(
            str(tex_path),
            "docx",
            outputfile=str(docx_path),
            extra_args=["--quiet"]
        )
        return True  # Success if no exception
    except Exception:
        return False  # Any failure yields False

def main():
    # Third person: Set up CLI arguments and run generation loop
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output_dir", default="pipeline")
    args = ap.parse_args()

    base = Path(args.output_dir)
    latex_dir = base / "latex"
    docx_dir = base / "docx"
    latex_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)

    profiles = json.loads(Path(args.profiles).read_text("utf-8"))
    jobs = json.loads(Path(args.jobs).read_text("utf-8"))
    job_descs = [j.get("description", str(j)) if isinstance(j, dict) else str(j) for j in jobs]

    manifest = []
    for i, profile in enumerate(profiles):
        # I pick a random job description for each profile
        jd = random.choice(job_descs)
        latex = generate_resume_latex(profile, jd)
        uid = uuid.uuid4().hex[:8]  # You ensure unique filenames
        tex_path = latex_dir / f"resume_{i}_{uid}.tex"
        docx_path = docx_dir / f"resume_{i}_{uid}.docx"
        ok = compile_latex_to_docx(latex, tex_path, docx_path)
        logging.info("%d %s", i, "OK" if ok else "FAIL")
        manifest.append({
            "profile_index": i,
            "tex": str(tex_path),
            "docx": str(docx_path) if ok else None,
            "compiled": ok,
            "job_description": jd
        })

    (base / "compiled_resume_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
