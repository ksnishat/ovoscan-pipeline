"""
OvoScan DVC Pipeline - Silver Layer
Preprocesses images: resize, normalize, augment.
"""
import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/bronze", help="Input images")
    parser.add_argument("--output", default="data/silver", help="Output images")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0
    for ext in IMAGE_EXTENSIONS:
        # rglob, not glob: the bronze layer preserves the class subdirectories,
        # so a non-recursive glob finds nothing.
        for f in src.rglob(f"*{ext}"):
            try:
                img = Image.open(f)
                img = img.resize((args.img_size, args.img_size))
                # Preserve the class subdirectory so the label survives.
                out = dst / f.relative_to(src)
                out = out.with_suffix(".jpg")
                out.parent.mkdir(parents=True, exist_ok=True)
                img.convert("RGB").save(out, quality=85)
                count += 1
            except Exception as e:
                errors += 1
                print(f"Error: {f.name} - {e}")

    if errors:
        print(f"Silver: {count} images preprocessed, {errors} failed")

    print(f"Silver: {count} images preprocessed to {args.img_size}x{args.img_size}")

if __name__ == "__main__":
    main()
