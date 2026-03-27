"""
reduce_overlayed_images.py

Randomly samples images from 'overlayed_images' and copies them to a new
'overlayed_images_reduced' folder.

Usage:
    python reduce_overlayed_images.py [--count N] [--ratio R] [--seed S]

Arguments:
    --count  N   Number of images to sample (default: 1000)
    --ratio  R   Fraction of images to sample, e.g. 0.1 for 10%
                 (overrides --count if provided)
    --seed   S   Random seed for reproducibility (default: 42)
"""

import argparse
import os
import random
import shutil


def get_image_files(folder: str) -> list[str]:
    """Return a sorted list of image file paths inside folder (non-recursive)."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in extensions
    ]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Reduce overlayed_images via random sampling."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of images to sample (default: 1000)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Fraction of total images to sample, e.g. 0.1 (overrides --count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--src",
        type=str,
        default="no-rain",
        help="Source folder (default: no_rain)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="no-rain_reduced",
        help="Destination folder (default: no_rain_reduced)",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    all_images = get_image_files(src_dir)
    total = len(all_images)

    if total == 0:
        print("No images found in source folder. Exiting.")
        return

    # Determine sample size
    if args.ratio is not None:
        if not (0 < args.ratio <= 1.0):
            raise ValueError("--ratio must be between 0 (exclusive) and 1 (inclusive).")
        sample_size = max(1, int(total * args.ratio))
    else:
        sample_size = min(args.count, total)

    print(f"Source      : {src_dir}")
    print(f"Destination : {dst_dir}")
    print(f"Total images: {total:,}")
    print(f"Sample size : {sample_size:,}")
    print(f"Seed        : {args.seed}")

    random.seed(args.seed)
    sampled = random.sample(all_images, sample_size)

    os.makedirs(dst_dir, exist_ok=True)

    for i, src_path in enumerate(sorted(sampled), 1):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
        if i % 100 == 0 or i == sample_size:
            print(f"  Copied {i:,}/{sample_size:,} ...", end="\r")

    print(f"\nDone! {sample_size:,} images copied to '{dst_dir}'.")


if __name__ == "__main__":
    main()
