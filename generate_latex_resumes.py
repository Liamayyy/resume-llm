#!/usr/bin/env python3
"""
Generate LaTeX resumes for every applicant profile
–––––––––––––––––––––––––––––––––––––––––––––––––––
• Reads applicant_profiles.json (same folder)
• Calls Galactica via HuggingFace Transformers
• Saves <first-last>.tex into build/latex/
• Uses plain‑text prompt ‑‑ *no JSON sent to the model*
"""
from pathlib import Path
import json
import textwrap
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build" / "latex"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

def _profile_to_plaintext(p: dict) -> str:
    """Convert one profile dict to readable plain‑text blocks."""
    edu_lines = "\n".join(f"  • {e['degree_type']} in {e['major']} — {e['college']}"
                          for e in p.get("education", []))
    job_lines = "\n".join(f"  • {j['job_title']} @ {j['company']}  ({j['years_in_role']} yrs)"
                          for j in p.get("job_history", []))
    skills = ", ".join(p.get("skills", []))
    return textwrap.dedent(f"""\
        Name: {p['first_name']} {p['last_name']}
        Email: {p['email']}
        Phone: {p['phone_number']}
        Location: {p['location']}

        Education:
        {edu_lines or '  • (none listed)'}

        Work History:
        {job_lines or '  • (none listed)'}

        Skills: {skills or '(none)'}
    """)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf", #or meta-llama/Llama-2-13b-chat-hf
                    help="HF model repo or local path")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    print("🔹 Loading tokenizer & model … (this may take a while)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok)

    profiles = json.loads(Path("applicant_profiles.json").read_text())

    for prof in profiles:
        plain = _profile_to_plaintext(prof)
        prompt = (
            "You are an expert technical résumé writer.\n"
            "Given the following applicant information, format a professional résumé "
            "in **LaTeX** suitable for PDF production (use the article class or "
            "moderncv—your choice).  The output must be valid compilable LaTeX and "
            "should not include any explanations—**only LaTeX code**.\n\n"
            "Applicant information ↓↓↓\n"
            f"{plain}\n\n"
            "LaTeX résumé:\n"
        )

        print(f"🖋  Generating LaTeX for {prof['first_name']} {prof['last_name']} …")
        out = pipe(prompt, max_new_tokens=args.max_new_tokens,
                   temperature=0.7, top_p=0.9, do_sample=True)[0]["generated_text"]

        # strip the prompt portion that the model echoes back, keep LaTeX only
        latex = out.split("LaTeX résumé:")[-1].lstrip()
        slug = f"{prof['first_name']}_{prof['last_name']}".lower().replace(" ", "_")
        tex_path = BUILD_DIR / f"{slug}.tex"
        tex_path.write_text(latex)
        print(f"   → {tex_path.relative_to(ROOT)} written.")

if __name__ == "__main__":
    main()
