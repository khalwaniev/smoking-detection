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
        for f in faces:
            emb = f.normed_embedding
            sims = np.dot(self.gallery.embeddings, emb)
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            if best_score >= 1 - config.RECOGNITION_COS_THRESHOLD:
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
