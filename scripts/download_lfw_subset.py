#!/usr/bin/env python
"""
Prepare a small subset of LFW images for demo gallery
(using an already extracted lfw-deepfunneled directory).
"""
import random, shutil
from pathlib import Path

BASE = Path("data/raw/lfw")
EXTRACT_DIR = BASE / "lfw-deepfunneled"   # this should already exist!
GALLERY_OUT = Path("data/processed/gallery_faces")
NUM_IDS = 10
IMAGES_PER_ID = 5
random.seed(42)

def main():
    if not EXTRACT_DIR.exists():
        print("ERROR: Folder not found:", EXTRACT_DIR)
        print("Please make sure lfw-deepfunneled is extracted in", BASE)
        return
    # Collect identities with at least IMAGES_PER_ID images
    ids = []
    for person_dir in sorted(EXTRACT_DIR.iterdir()):
        imgs = list(person_dir.glob("*.jpg"))
        if len(imgs) >= IMAGES_PER_ID:
            ids.append((person_dir.name, imgs))
    print(f"Total identities with >= {IMAGES_PER_ID} images:", len(ids))
    if len(ids) < NUM_IDS:
        print(f"ERROR: Not enough identities with at least {IMAGES_PER_ID} images!")
        return
    random.shuffle(ids)
    chosen = ids[:NUM_IDS]
    # Clear old gallery subset if any
    for c in chosen:
        out_dir = GALLERY_OUT / c[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        # copy first IMAGES_PER_ID images
        for i, img in enumerate(c[1][:IMAGES_PER_ID]):
            shutil.copy(img, out_dir / f"{c[0]}_{i}.jpg")
    print(f"Prepared gallery subset: {NUM_IDS} identities × {IMAGES_PER_ID} images.")
    print("Output folder:", GALLERY_OUT)

if __name__ == "__main__":
    main()
