import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    """
    Returns cosine similarity between two embedding vectors.
    1.0 = identical, 0.0 = unrelated.
    """
    a = np.array(embedding_a).flatten()
    b = np.array(embedding_b).flatten()

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)