#!/usr/bin/env python
"""
Generate synthetic feature vectors to mimic evidence signals for smoking vs non-smoking.
Outputs CSV: outputs/reports/synthetic_features.csv
"""
import csv, random, math
from pathlib import Path

N_POS = 300
N_NEG = 300
OUT_PATH = Path("outputs/reports")
OUT_PATH.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_PATH / "synthetic_features.csv"

FIELDS = [
    "label","person_conf","hand_item_prob","cig_prob","vape_prob",
    "h2m_prob","smoke_prob","emission_prob","action_prob","firecracker_prob"
]

def pos_sample():
    # Active smoking: high person, cig OR vape, h2m, smoke, emission, action
    return dict(
        person_conf=random.uniform(0.85,1.0),
        hand_item_prob=random.uniform(0.4,0.9),
        cig_prob=random.uniform(0.5,0.95),
        vape_prob=random.uniform(0.0,0.3), # mostly cigarette in this synthetic; will mix
        h2m_prob=random.uniform(0.5,1.0),
        smoke_prob=random.uniform(0.6,1.0),
        emission_prob=random.uniform(0.5,0.95),
        action_prob=random.uniform(0.5,0.95),
        firecracker_prob=random.uniform(0.0,0.15)
    )

def neg_sample():
    mode = random.choice(["person_only","hand_obj","firecracker","random"])
    if mode == "person_only":
        return dict(
            person_conf=random.uniform(0.7,1.0),
            hand_item_prob=random.uniform(0.0,0.3),
            cig_prob=random.uniform(0.0,0.15),
            vape_prob=random.uniform(0.0,0.15),
            h2m_prob=random.uniform(0.0,0.25),
            smoke_prob=random.uniform(0.0,0.2),
            emission_prob=random.uniform(0.0,0.2),
            action_prob=random.uniform(0.0,0.3),
            firecracker_prob=random.uniform(0.0,0.05)
        )
    if mode == "hand_obj":
        return dict(
            person_conf=random.uniform(0.8,1.0),
            hand_item_prob=random.uniform(0.4,0.8),
            cig_prob=random.uniform(0.0,0.25),
            vape_prob=random.uniform(0.0,0.25),
            h2m_prob=random.uniform(0.0,0.35),
            smoke_prob=random.uniform(0.0,0.15),
            emission_prob=random.uniform(0.0,0.15),
            action_prob=random.uniform(0.0,0.35),
            firecracker_prob=random.uniform(0.0,0.05)
        )
    if mode == "firecracker":
        return dict(
            person_conf=random.uniform(0.3,0.9),
            hand_item_prob=random.uniform(0.0,0.2),
            cig_prob=random.uniform(0.0,0.2),
            vape_prob=random.uniform(0.0,0.1),
            h2m_prob=random.uniform(0.0,0.15),
            smoke_prob=random.uniform(0.4,0.8),  # plume confusion
            emission_prob=random.uniform(0.0,0.3),
            action_prob=random.uniform(0.0,0.25),
            firecracker_prob=random.uniform(0.5,0.95)
        )
    # random noise negative
    return dict(
        person_conf=random.uniform(0.0,1.0),
        hand_item_prob=random.uniform(0.0,0.5),
        cig_prob=random.uniform(0.0,0.2),
        vape_prob=random.uniform(0.0,0.2),
        h2m_prob=random.uniform(0.0,0.3),
        smoke_prob=random.uniform(0.0,0.3),
        emission_prob=random.uniform(0.0,0.3),
        action_prob=random.uniform(0.0,0.35),
        firecracker_prob=random.uniform(0.0,0.2)
    )

with OUT_FILE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    for _ in range(N_POS):
        d = pos_sample()
        d["label"] = 1
        writer.writerow(d)
    for _ in range(N_NEG):
        d = neg_sample()
        d["label"] = 0
        writer.writerow(d)

print(f"Saved synthetic feature dataset -> {OUT_FILE}")
