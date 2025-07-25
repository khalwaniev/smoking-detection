from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES = DATA_DIR / "raw" / "images"
RAW_VIDEOS = DATA_DIR / "raw" / "videos"
GALLERY_DIR = DATA_DIR / "processed" / "gallery_faces"
EMBEDDINGS_FILE = DATA_DIR / "processed" / "gallery_embeddings.npy"
METADATA_FILE = DATA_DIR / "processed" / "gallery_metadata.csv"
OUTPUT_LOGS = PROJECT_ROOT / "outputs" / "logs"
DETECTIONS_DIR = PROJECT_ROOT / "outputs" / "detections"

for p in [RAW_IMAGES, RAW_VIDEOS, GALLERY_DIR, OUTPUT_LOGS, DETECTIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Thresholds (tune later)
FACE_DET_SCORE_THRESH = 0.9
RECOGNITION_COS_THRESHOLD = 0.30  # updated from threshold_analysis.py
SMOKING_CONF_THRESHOLD = 0.8

