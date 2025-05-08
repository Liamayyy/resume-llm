import argparse
import subprocess
import sys
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(message)s")
DIR = Path(__file__).resolve().parent  # I locate scripts relative to wrapper

def _run(script, *args):
    # You execute each sub-script in turn
    cmd = [sys.executable, str(DIR / script), *args]
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output_dir", default="pipeline")
    args = ap.parse_args()

    _run(
        "generate_resumes.py",
        "--profiles", args.profiles,
        "--jobs", args.jobs,
        "--output_dir", args.output_dir
    )
    # Third person: Now run the ATS scorer
    manifest = str(Path(args.output_dir) / "compiled_resume_manifest.json")
    _run(
        "ats_scoring.py",
        "--manifest", manifest,
        "--output_dir", str(Path(args.output_dir) / "ats_results")
    )
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
