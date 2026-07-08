import cv2
import numpy as np
from keras_facenet import FaceNet


class EmbeddingGenerator:
    def __init__(self):
        self.model = FaceNet()

    def get_embedding(self, face_image):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (160, 160))
        face_image = np.expand_dims(face_image, axis=0)
        embedding = self.model.embeddings(face_image)
        return embedding[0]