import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from app.detector import FaceDetector
from config.settings import MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "face_cnn.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "class_labels.json")
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.85  # below this, treat as Unknown


class CNNFaceRecognizer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run scripts/train_cnn_classifier.py first."
            )

        self.detector = FaceDetector()
        self.model = load_model(MODEL_PATH)

        with open(LABELS_PATH, "r") as f:
            self.class_names = json.load(f)

    def detect_faces(self, frame):
        return self.detector.detect_faces(frame)

    def recognize(self, frame, box):
        face_crop = self.detector.crop_face(frame, box)
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = cv2.resize(face_crop, IMAGE_SIZE)
        face_crop = face_crop.astype("float32") / 255.0
        face_crop = np.expand_dims(face_crop, axis=0)

        predictions = self.model.predict(face_crop, verbose=0)[0]
        best_index = np.argmax(predictions)
        confidence = float(predictions[best_index])

        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown", confidence * 100

        return self.class_names[best_index], confidence * 100

    def refresh_known_faces(self):
        # No-op for compatibility with FaceRecognizer's interface —
        # this model needs retraining, not a hot reload, to learn new faces.
        pass