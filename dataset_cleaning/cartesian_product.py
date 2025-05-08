#!/usr/bin/env python3
"""
create_cartesian_text_dataset.py

Loads applicant profiles and job postings, computes their Cartesian product,
and produces a Hugging Face Dataset of plain-text examples,
where each example is one string containing both an applicant and a job.
"""

import json
import itertools
from pathlib import Path

from datasets import Dataset  # pip install datasets


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_list(title: str, items: list, fields: list) -> list:
    """
    Helper to format a list of dicts. Returns a list of strings.
    title: heading for the section (e.g., 'Education' or 'Job History')
    items: list of dicts
    fields: list of tuples (key, pretty_name)
    """
    lines = [f"{title}:"]
    if not items:
        lines.append("None")
        return lines
    for entry in items:
        parts = []
        for key, pretty in fields:
            value = entry.get(key, "N/A")
            parts.append(f"{pretty}: {value}")
        lines.append("; ".join(parts))
    return lines


def format_entry(applicant: dict, job: dict) -> str:
    """
    Turn one applicant dict and one job dict into a single text string.
    """
    lines = []
    lines.append("=== Applicant Profile ===")
    # Basic fields
    for key in ("First Name", "Last Name", "Email", "Phone Number", "Location"):  # adjust if keys differ
        val = applicant.get(key.lower().replace(" ", "_"), None)
        if val:
            lines.append(f"{key}: {val}")

    # Education list
    edu_fields = [
        ("degree_type", "Degree"),
        ("major", "Major"),
        ("college", "College"),
    ]
    lines.extend(format_list("Education", applicant.get("education", []), edu_fields))

    # Job history list
    job_fields = [
        ("job_title", "Title"),
        ("company", "Company"),
        ("years_in_role", "Years in Role"),
    ]
    lines.extend(format_list("Job History", applicant.get("job_history", []), job_fields))

    # Skills list
    lines.append("Skills:")
    skills = applicant.get("skills", [])
    if skills:
        # comma-separated
        lines.append(", ".join(skills))
    else:
        lines.append("None")

    lines.append("")  # blank line before job
    
    # Job posting
    lines.append("=== Job Posting ===")
    # About Us
    if job.get("about_us"):
        lines.append(f"About Us: {job['about_us']}")

    # Description
    if job.get("description"):
        lines.append(f"Description: {job['description']}")

    # Requirements
    if job.get("requirements"):
        lines.append("Requirements:")
        # split on common delimiters
        reqs = job["requirements"].split(";") if isinstance(job["requirements"], str) else job["requirements"]
        if isinstance(reqs, list):
            for r in reqs:
                lines.append(f"- {r.strip()}")
        else:
            lines.append(f"{job['requirements']}")

    return "\n".join(lines)


def main():
    # adjust these paths if needed
    profiles_path = Path("./datasets/applicant_profiles.json")
    jobs_path     = Path("./datasets/filtered_jobs.json")

    applicants = load_json(profiles_path)
    jobs       = load_json(jobs_path)

    examples = []
    for applicant, job in itertools.product(applicants, jobs):
        text = format_entry(applicant, job)
        examples.append({"text": text})

    # build HF dataset
    ds = Dataset.from_list(examples)
    print(ds)         # schema & size
    print(ds[0]["text"])  # inspect first entry

    # save to disk for later loading in your RLHF script
    out_dir = Path("cartesian_text_dataset")
    ds.save_to_disk(out_dir)
    print(f"Saved plain‑text Cartesian dataset to {out_dir}/")

    # (Optional) push to Hub:
    ds.push_to_hub("Liamayyy/applicant-job-plaintext")

if __name__ == "__main__":
    main()
