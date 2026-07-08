import os
import pickle
import cv2

from app.detector import FaceDetector
from app.embedding import EmbeddingGenerator
from utils.image_quality import check_quality
from utils.helper import cosine_similarity
from utils.logger import logger
from config.settings import EMBEDDINGS_FILE, DATASET_DIR, SIMILARITY_THRESHOLD


class RegistrationManager:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = EmbeddingGenerator()
        self.embeddings_db = self._load_embeddings()

    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(self.embeddings_db, f)

    def is_duplicate(self, new_embedding, exclude_name=None):
        """Returns (True, matched_name) if new_embedding matches an existing student."""
        for name, embeddings in self.embeddings_db.items():
            if name == exclude_name:
                continue
            for emb in embeddings:
                if cosine_similarity(new_embedding, emb) >= SIMILARITY_THRESHOLD:
                    return True, name
        return False, None

    def register_from_folder(self, student_name, folder_path):
        """Builds embeddings for a student from a folder of existing images."""
        if not os.path.isdir(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return 0

        collected_embeddings = []

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            faces = self.detector.detect_faces(image)
            if len(faces) == 0:
                logger.info(f"No face found in {filename}")
                continue

            # Use the largest face if multiple are detected
            box = max(faces, key=lambda b: b[2] * b[3])
            face_crop = self.detector.crop_face(image, box)

            good, reason = check_quality(face_crop, box)
            if not good:
                logger.info(f"Skipped {filename}: {reason}")
                continue

            embedding = self.embedder.get_embedding(face_crop)
            collected_embeddings.append(embedding)

        if not collected_embeddings:
            logger.warning(f"No usable faces found for {student_name}")
            return 0

        self.embeddings_db[student_name] = collected_embeddings
        self._save_embeddings()

        logger.info(f"Registered {student_name} with {len(collected_embeddings)} embeddings.")
        return len(collected_embeddings)

    def register_all_from_dataset(self):
        """Scans dataset/<student_name>/ and builds embeddings for every folder found."""
        if not os.path.isdir(DATASET_DIR):
            logger.warning("Dataset directory not found.")
            return

        for student_name in os.listdir(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, student_name)
            if os.path.isdir(folder_path):
                self.register_from_folder(student_name, folder_path)