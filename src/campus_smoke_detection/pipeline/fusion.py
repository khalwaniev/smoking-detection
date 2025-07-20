import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FusionResult:
    raw_score: float
    probability: float
    label: str
    contributions: Dict[str, float]

def load_fusion_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def fuse_features(features: Dict[str, float], cfg: dict) -> FusionResult:
    weights = cfg["weights"]
    bias = cfg.get("bias", 0.0)
    clamp_min = cfg.get("clamp_min", 0.0)
    clamp_max = cfg.get("clamp_max", 1.0)
    contributions = {}
    score = bias
    for k, w in weights.items():
        v = float(features.get(k, 0.0))
        contrib = w * v
        contributions[k] = contrib
        score += contrib
    score = float(np.clip(score, clamp_min, clamp_max))
    # For now, probability = score (identity). Later we can calibrate.
    prob = score
    th = cfg.get("thresholds", {})
    if prob < th.get("unlikely", 0.25):
        label = "UNLIKELY"
    elif prob < th.get("low", 0.50):
        label = "LOW"
    elif prob < th.get("moderate", 0.70):
        label = "MODERATE"
    elif prob < th.get("high", 0.85):
        label = "HIGH"
    else:
        label = "CONFIRMED"
    return FusionResult(raw_score=score, probability=prob, label=label, contributions=contributions)
