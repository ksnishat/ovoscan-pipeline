"""
OvoScan DVC Pipeline - Bronze Layer
Extracts image data for defect classification.
"""
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw", help="Source images directory")
    parser.add_argument("--output", default="data/bronze", help="Output directory")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    counts = {}
    copied = 0
    for ext in IMAGE_EXTENSIONS:
        files = list(src.rglob(f"*{ext}"))
        counts[ext] = len(files)
        for f in files:
            # Preserve the directory structure relative to the source root so
            # files with the same name in different class folders do not collide.
            dst_file = dst / f.relative_to(src)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_file)
            copied += 1
            logger.info(f"Bronze: {f.relative_to(src)}")

    print(
        f"Extracted: JPG={counts.get('.jpg', 0)}, "
        f"PNG={counts.get('.png', 0)}, "
        f"JPEG={counts.get('.jpeg', 0)} ({copied} files copied to {dst})"
    )


if __name__ == "__main__":
    main()
