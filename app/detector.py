import cv2
from config.settings import (
    HAAR_CASCADE,
    DETECT_SCALE_FACTOR,
    DETECT_MIN_NEIGHBORS,
    DETECT_MIN_SIZE,
)


class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        return faces  # list of (x, y, w, h)

    def crop_face(self, frame, box):
        x, y, w, h = box
        return frame[y:y + h, x:x + w]