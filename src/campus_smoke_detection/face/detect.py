import cv2
from insightface.app import FaceAnalysis
from loguru import logger

class FaceDetector:
    def __init__(self, providers=None, det_size=(640,640)):
        # ... constructor code ...
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, image_bgr):
        faces = self.app.get(image_bgr)
        return faces
