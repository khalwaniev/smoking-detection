import numpy as np
from loguru import logger
from .detect import FaceDetector
from .gallery import FaceGallery
from .. import config

class FaceRecognizer:
    def __init__(self):
        self.detector = FaceDetector()
        self.gallery = FaceGallery()
        self.gallery.load()

    def recognize_in_image(self, image_bgr):
        faces = self.detector.detect(image_bgr)
        results = []
        if len(faces) == 0:
            logger.info("No faces detected.")
            return results
        for f in faces:
            emb = f.normed_embedding
            if emb is None or len(emb) == 0:
                logger.warning("Face detected without embedding. Skipping.")
                continue
            sims = np.dot(self.gallery.embeddings, emb)
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            # Threshold logic: accept if above empirical cosine similarity
            if best_score >= config.RECOGNITION_COS_THRESHOLD:
                person_id = self.gallery.meta.iloc[best_idx].person_id
            else:
                person_id = "UNKNOWN"
            results.append({
                "bbox": f.bbox.tolist(),
                "det_score": float(f.det_score),
                "id": person_id,
                "similarity": float(best_score)
            })
        return results
