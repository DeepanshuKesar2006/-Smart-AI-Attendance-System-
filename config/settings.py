import os
import cv2

# -----------------------------
# Base paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "processed_dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")

# -----------------------------
# Face detection
# -----------------------------
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECT_SCALE_FACTOR = 1.2
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE = (80, 80)

# -----------------------------
# Recognition
# -----------------------------
EMBEDDING_SIZE = 128
SIMILARITY_THRESHOLD = 0.70   # cosine similarity above this = match
ATTENDANCE_COOLDOWN_SECONDS = 60 * 60  # 1 hour before re-marking same person

# -----------------------------
# Camera
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# -----------------------------
# UI / display
# -----------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# -----------------------------
# Ensure required directories exist
# -----------------------------
for directory in [
    DATASET_DIR,
    PROCESSED_DATASET_DIR,
    EMBEDDINGS_DIR,
    DATABASE_DIR,
    LOGS_DIR,
    MODELS_DIR,
]:
    os.makedirs(directory, exist_ok=True)