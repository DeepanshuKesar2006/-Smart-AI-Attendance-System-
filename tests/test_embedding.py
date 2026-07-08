import cv2

from utils.embedding_utils import FaceEmbedding

image = cv2.imread(
    "processed_dataset/Deepanshu/0.jpg"
)

model = FaceEmbedding()

embedding = model.get_embedding(image)

print(embedding.shape)