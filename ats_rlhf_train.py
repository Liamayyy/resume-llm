#!/usr/bin/env python3
"""
ats_rlhf_train.py  –  PPO-based RLHF fine-tuner (TRL ≥0.12.0)
=============================================================
Uses ATS score as reward to fine-tune a resume-writing LLM.

Usage:
python ats_rlhf_train.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --profiles applicant_profiles.json \
    --job job_descriptions/front_end_eng.pdf \
    --epochs 3 --batch-size 4
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, re, tempfile, textwrap

import torch
import torch.nn as nn
import pypandoc, pdfplumber, docx

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
)
from trl import PPOTrainer, PPOConfig, create_reference_model

from ats_score import ats_score


# ─────── Dummy reward model ────────────────────────────────
class DummyRewardModel(nn.Module):
    """No-op reward model; real rewards come via ppo.step()"""
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = (
            input_ids.size(0)
            if input_ids is not None
            else (attention_mask.size(0) if attention_mask is not None else 1)
        )
        return torch.zeros(batch_size, device="cpu")


# ─────── Helpers ─────────────────────────────────────────
def _extract_job(path: Path, max_chars: int = 4000) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        with pdfplumber.open(path) as pdf:
            txt = "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif suf == ".docx":
        d = docx.Document(path)
        txt = "\n".join(p.text for p in d.paragraphs)
    else:
        raise ValueError("JD must be .pdf or .docx")
    return re.sub(r"\s+", " ", txt).strip()[:max_chars]


def _profile_txt(p: dict) -> str:
    edu = "\n".join(
        f"  • {e['degree_type']} in {e['major']} — {e['college']}"
        for e in p.get("education", [])
    ) or "  • (none listed)"
    jobs = "\n".join(
        f"  • {j['job_title']} @ {j['company']} ({j['years_in_role']} yrs)"
        for j in p.get("job_history", [])
    ) or "  • (none listed)"
    skills = ", ".join(p.get("skills", [])) or "(none)"
    return textwrap.dedent(f"""\
        Name: {p['first_name']} {p['last_name']}
        Email: {p['email']}
        Phone: {p['phone_number']}
        Location: {p['location']}

        Education:
        {edu}

        Work History:
        {jobs}

        Skills: {skills}
    """)


def _latex_to_docx(latex: str) -> Path:
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "resume.tex"
        tex.write_text(latex)
        out = Path(td) / "resume.docx"
        pypandoc.convert_file(str(tex), "docx", outputfile=str(out))
        return out


def _reward(latex: str, jd_path: Path) -> float:
    try:
        docx_path = _latex_to_docx(latex)
        return ats_score(str(docx_path), str(jd_path)) / 100.0
    except Exception as e:
        print("⚠️ reward error:", e)
        return 0.0


# ─────── Main ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--profiles",   default="applicant_profiles.json")
    ap.add_argument("--job",        required=True, type=Path)
    ap.add_argument("--epochs",     type=int,   default=1)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--save-dir",   default="ckpts")
    ap.add_argument("--save-every", type=int,   default=500)
    args = ap.parse_args()

    jd_text  = _extract_job(args.job)
    profiles = json.loads(Path(args.profiles).read_text())

    # 1) Policy + Tokenizer
    tok   = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 2) Reference copy for KL
    ref_model = create_reference_model(policy_model)

    # 3) Value model (proper base_model_prefix & backbone)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1
    ).to(policy_model.device)

    # 4) Generation config for sampling
    policy_model.generation_config = GenerationConfig.from_pretrained(args.model)

    # 5) Dummy reward module
    reward_model = DummyRewardModel()

    # 6) Dummy dataset
    train_dataset = [{"query": ""}] * args.batch_size

    # 7) Build PPOTrainer
    ppo = PPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        args=PPOConfig(batch_size=args.batch_size, learning_rate=1e-6),
        processing_class=tok,
    )

    # 8) Training loop
    step   = 0
    ck_dir = Path(args.save_dir); ck_dir.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        for prof in profiles:
            prompt = (
                "You are an expert technical résumé writer.\n"
                "Generate a *tailored* résumé in **LaTeX** and "
                "provide **only** the LaTeX code.\n\n"
                f"Applicant information:\n{_profile_txt(prof)}\n\n"
                f"Job description:\n{jd_text}\n\n"
                "LaTeX resume:\n"
            )

            batch = tok(prompt, return_tensors="pt").to(policy_model.device)
            q_ids = batch["input_ids"]
            r_ids = ppo.generate(q_ids, max_new_tokens=512,
                                 temperature=0.7, top_p=0.9)

            responses = tok.batch_decode(r_ids, skip_special_tokens=True)
            rewards = [
                _reward(r.split("LaTeX resume:")[-1].lstrip(), args.job)
                for r in responses
            ]

            ppo.step(q_ids, r_ids, rewards)
            step += 1

            if step % args.save_every == 0:
                ck = ck_dir / f"step{step:06d}"
                policy_model.save_pretrained(ck)
                tok.save_pretrained(ck)
                print(f"💾 checkpoint saved: {ck}")

    # Final checkpoint
    final_ck = ck_dir / "final"
    policy_model.save_pretrained(final_ck)
    tok.save_pretrained(final_ck)
    print(f"🎉 training complete → {final_ck}")


if __name__ == "__main__":
    main()
