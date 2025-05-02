#!/usr/bin/env python3
"""
Run ats_scoring.py on every DOCX resume against a single JD PDF.
Logs verbose output into build/ats_results.log
"""
import subprocess
from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parent
DOCX_DIR = ROOT / "build" / "docx"
ATS = ROOT / "ats_scoring.py"
JOB_DESC = ROOT / "job_descriptions" / "job_desc_front_end_engineer.pdf"
LOG = ROOT / "build" / "ats_results.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def main():
    with LOG.open("w") as fh:
        fh.write(f"ATS run {datetime.datetime.now()}\n")
        fh.write(f"Job description: {JOB_DESC}\n\n")

        for cv in sorted(DOCX_DIR.glob("*.docx")):
            cmd = ["python", str(ATS), "-j", str(JOB_DESC), "-r", str(cv)]
            fh.write(f"$ {' '.join(cmd)}\n")
            print(f"⚙️  Scoring {cv.name} …")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            fh.write(proc.stdout)
            if proc.stderr:
                fh.write("\n[stderr]\n" + proc.stderr)
            fh.write("\n" + "-"*60 + "\n")

    print(f"📄 All results saved to {LOG.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
