#!/usr/bin/env python
"""
Compute genuine vs impostor similarity distributions from gallery images.
We treat each image embedding; pairs same identity = genuine, different = impostor.
"""
import numpy as np, itertools, pandas as pd
from campus_smoke_detection.face.gallery import FaceGallery
from campus_smoke_detection import config

def main():
    g = FaceGallery()
    g.load()
    embs = g.embeddings
    meta = g.meta
    pairs_genuine = []
    pairs_impostor = []
    # Precompute normalized (ArcFace already normed)
    for (i,a),(j,b) in itertools.combinations(enumerate(embs),2):
        sim = float(np.dot(a,b))
        same = meta.iloc[i].person_id == meta.iloc[j].person_id
        if same: pairs_genuine.append(sim)
        else: pairs_impostor.append(sim)
    print(f"Genuine pairs: {len(pairs_genuine)}, Impostor pairs: {len(pairs_impostor)}")
    # Simple stats
    import statistics
    print("Genuine mean:", statistics.mean(pairs_genuine))
    print("Impostor mean:", statistics.mean(pairs_impostor))
    # Suggest threshold: choose value with low false accept
    # e.g., pick 0.55 or dynamic
    # Write CSV for later plotting
    import csv
    with open("outputs/reports/similarity_pairs.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["type","similarity"])
        for s in pairs_genuine: w.writerow(["genuine",s])
        for s in pairs_impostor: w.writerow(["impostor",s])
    # Quick threshold scan
    thresholds = np.linspace(0.3,0.9,25)
    best = None
    for t in thresholds:
        tp = sum(s>=t for s in pairs_genuine)
        fn = sum(s< t for s in pairs_genuine)
        fp = sum(s>=t for s in pairs_impostor)
        tn = sum(s< t for s in pairs_impostor)
        far = fp / (fp+tn) if (fp+tn) > 0 else 0.0
        tar = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        if best is None or (tar - far) > best[0]:
            best = (tar - far, t, tar, far)
    print(f"Suggested threshold: {best[1]:.3f} (TAR={best[2]:.3f}, FAR={best[3]:.3f})")

if __name__ == "__main__":
    main()
