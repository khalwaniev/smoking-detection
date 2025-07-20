import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from .detect import FaceDetector
from .. import config

class FaceGallery:
    def __init__(self):
        self.embeddings = None
        self.meta = None

    def build(self, gallery_root=config.GALLERY_DIR):
        detector = FaceDetector()
        rows = []
        embs = []
        for person_dir in sorted(Path(gallery_root).glob("*")):
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            for img_path in person_dir.glob("*.jpg"):
                import cv2
                img = cv2.imread(str(img_path))
                faces = detector.detect(img)
                if len(faces) != 1:
                    logger.warning(f"{img_path} -> expected 1 face, found {len(faces)}; skipping.")
                    continue
                emb = faces[0].normed_embedding
                embs.append(emb)
                rows.append({"person_id": person_id, "img_path": str(img_path)})
        self.embeddings = np.vstack(embs)
        self.meta = pd.DataFrame(rows)
        np.save(config.EMBEDDINGS_FILE, self.embeddings)
        self.meta.to_csv(config.METADATA_FILE, index=False)
        logger.info(f"Gallery built: {self.embeddings.shape[0]} embeddings.")

    def load(self):
        self.embeddings = np.load(config.EMBEDDINGS_FILE)
        import pandas as pd
        self.meta = pd.read_csv(config.METADATA_FILE)
