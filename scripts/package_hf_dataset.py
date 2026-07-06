"""Package the Hugging Face dataset repo from local files. Run on Kaggle only."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="../../hugging_face/dataset_repo")
    parser.add_argument("--dst", default="/kaggle/working/hf_dataset_repo")
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Dataset package copied to {dst}")


if __name__ == "__main__":
    main()
