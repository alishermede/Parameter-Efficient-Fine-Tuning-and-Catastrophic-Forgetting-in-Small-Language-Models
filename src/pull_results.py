"""Download run logs that Kaggle sessions pushed to the private Hub repo(s) into results/.

  python src/pull_results.py --repos user1/rmt-peft-ckpts user2/rmt-peft-ckpts user3/rmt-peft-ckpts

Only `results/**` is downloaded (logs, metrics, predictions), never the checkpoints.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--dest", default=str(RESULTS_DIR / "raw"))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for repo in args.repos:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(repo_id=repo, allow_patterns=["results/*", "results/**"], local_dir=tmp)
            src = Path(tmp) / "results"
            if not src.exists():
                print(f"{repo}: no results yet")
                continue
            for run_dir in sorted(p for p in src.iterdir() if p.is_dir()):
                target = (RESULTS_DIR / "lr_search" if run_dir.name.startswith("lrsearch_") else dest) / run_dir.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(run_dir, target)
                print(f"{repo}: {run_dir.name} -> {target}")


if __name__ == "__main__":
    main()
