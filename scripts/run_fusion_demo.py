#!/usr/bin/env python
import csv
from pathlib import Path
from campus_smoke_detection.pipeline.fusion import load_fusion_config, fuse_features

CFG_PATH = "configs/fusion_weights.yaml"
DATA_FILE = "outputs/reports/synthetic_features.csv"

def main():
    cfg = load_fusion_config(CFG_PATH)
    positives = 0
    total = 0
    counts = {"UNLIKELY":0,"LOW":0,"MODERATE":0,"HIGH":0,"CONFIRMED":0}
    samples_to_print = 5
    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = {k: float(row[k]) for k in row if k not in ("label")}
            result = fuse_features(features, cfg)
            total += 1
            counts[result.label]+=1
            if samples_to_print>0:
                print("Sample:")
                print(" label_gt=", row["label"], "prob=", f"{result.probability:.3f}", "label_pred=", result.label)
                print(" contributions=", {k: round(v,3) for k,v in result.contributions.items()})
                print("---")
                samples_to_print -= 1
            if result.label in ("HIGH","CONFIRMED"):
                positives += 1
    print("Distribution:", counts)
    print(f"Total {total} samples processed.")

if __name__ == "__main__":
    main()
