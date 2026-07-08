from app.detector import FaceDetector
from app.embedding import EmbeddingGenerator
from app.matcher import FaceMatcher


class FaceRecognizer:
    """
    Single entry point for recognition. This is the ONLY interface
    the rest of the app (run.py, dashboard.py) should ever call.
    """

    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = EmbeddingGenerator()
        self.matcher = FaceMatcher()

    def detect_faces(self, frame):
        """Returns list of (x, y, w, h) boxes."""
        return self.detector.detect_faces(frame)

    def recognize(self, frame, box):
        """
        Takes a full frame and one face box.
        Returns (name, confidence_percent).
        """
        face_crop = self.detector.crop_face(frame, box)
        embedding = self.embedder.get_embedding(face_crop)
        name, similarity = self.matcher.match(embedding)
        confidence = similarity * 100
        return name, confidence

    def refresh_known_faces(self):
        """Call after new students are registered."""
        self.matcher.reload()