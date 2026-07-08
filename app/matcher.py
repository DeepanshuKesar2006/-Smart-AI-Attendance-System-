import os
import pickle

from utils.helper import cosine_similarity
from utils.logger import logger
from config.settings import EMBEDDINGS_FILE, SIMILARITY_THRESHOLD


class FaceMatcher:
    def __init__(self):
        self.embeddings_db = self._load_embeddings()

    def _load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                return pickle.load(f)
        logger.warning("No embeddings file found. Recognition will always return Unknown.")
        return {}

    def reload(self):
        """Call this after registering a new student so matches stay up to date."""
        self.embeddings_db = self._load_embeddings()

    def match(self, query_embedding):
        """
        Compares query_embedding against every stored embedding.
        Returns (name, similarity) — name is "Unknown" if nothing clears the threshold.
        """
        best_name = "Unknown"
        best_score = 0.0

        for name, embeddings in self.embeddings_db.items():
            for emb in embeddings:
                score = cosine_similarity(query_embedding, emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < SIMILARITY_THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score