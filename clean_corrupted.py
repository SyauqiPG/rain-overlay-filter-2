import os
from pathlib import Path
from PIL import Image


def scan_and_delete_corrupted(folder, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Scan a folder for corrupted image files and delete them.

    Args:
        folder: Path to folder to scan
        extensions: Tuple of file extensions to check
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in extensions]
    total = len(image_paths)
    print(f"Scanning {total} images in '{folder}'...\n")

    corrupted = []
    for i, path in enumerate(image_paths, 1):
        if i % 1000 == 0:
            print(f"  Checked {i}/{total}...")
        try:
            with Image.open(path) as img:
                img.verify()  # Verify without fully loading
        except Exception:
            corrupted.append(path)

    if not corrupted:
        print("No corrupted images found.")
        return

    print(f"\nFound {len(corrupted)} corrupted file(s):")
    for p in corrupted:
        print(f"  {p.name}")

    confirm = input(f"\nDelete all {len(corrupted)} corrupted file(s)? [y/N]: ").strip().lower()
    if confirm == 'y':
        for p in corrupted:
            p.unlink()
            print(f"Deleted: {p.name}")
        print(f"\nDone. Deleted {len(corrupted)} corrupted file(s).")
    else:
        print("Aborted. No files deleted.")


if __name__ == '__main__':
    scan_and_delete_corrupted('overlayed_images')
