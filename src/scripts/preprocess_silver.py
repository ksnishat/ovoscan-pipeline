"""
OvoScan DVC Pipeline - Silver Layer
Preprocesses images: resize, normalize, augment.
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

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
    for ext in [".jpg", ".png", ".jpeg"]:
        for f in src.glob(f"*{ext}"):
            try:
                img = Image.open(f)
                img = img.resize((args.img_size, args.img_size))
                # Save to silver
                out = dst / f"{f.stem}.jpg"
                img.convert("RGB").save(out, quality=85)
                count += 1
            except Exception as e:
                print(f"Error: {f.name} - {e}")
    
    print(f"Silver: {count} images preprocessed to {args.img_size}x{args.img_size}")

if __name__ == "__main__":
    main()
