#!/usr/bin/env python
import sys
import cv2
from campus_smoke_detection.face.recognize import FaceRecognizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_recognition.py <image_path>")
        return
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image:", img_path)
        return
    rec = FaceRecognizer()
    results = rec.recognize_in_image(img)
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
