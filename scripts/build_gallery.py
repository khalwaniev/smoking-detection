#!/usr/bin/env python
import sys
from campus_smoke_detection.face.gallery import FaceGallery
from campus_smoke_detection import config
from pathlib import Path

def debug_build_gallery():
    gallery_root = config.GALLERY_DIR
    print("GALLERY ROOT BEING USED:", gallery_root)
    from campus_smoke_detection.face.detect import FaceDetector
    detector = FaceDetector()
    rows = []
    embs = []
    for person_dir in sorted(Path(gallery_root).glob("*")):
        print("PERSON DIR:", person_dir)
        if not person_dir.is_dir():
            print(f"SKIPPING (not a dir): {person_dir}")
            continue
        person_id = person_dir.name
        for img_path in person_dir.glob("*.jpg"):
            print("PROCESSING IMAGE:", img_path)
            import cv2
            img = cv2.imread(str(img_path))
            print("IMAGE SHAPE:", img.shape if img is not None else None)
            if img is None:
                print(f"FAILED TO READ IMAGE: {img_path}")
                continue
            faces = detector.detect(img)
            print("FACES FOUND:", len(faces))
            if len(faces) != 1:
                print(f"SKIPPING {img_path}: found {len(faces)} faces")
                continue
            emb = faces[0].normed_embedding
            embs.append(emb)
            rows.append({"person_id": person_id, "img_path": str(img_path)})
    print(f"TOTAL IMAGES WITH EMBEDDINGS: {len(embs)}")
    if len(embs) == 0:
        print("NO EMBEDDINGS COLLECTED. Exiting.")
        sys.exit(1)
    import numpy as np
    import pandas as pd
    np.save(config.EMBEDDINGS_FILE, np.vstack(embs))
    pd.DataFrame(rows).to_csv(config.METADATA_FILE, index=False)
    print(f"Gallery built: {len(embs)} embeddings.")

if __name__ == "__main__":
    debug_build_gallery()
