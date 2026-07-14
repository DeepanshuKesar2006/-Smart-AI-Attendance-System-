import sys
import os
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from app.cnn_model import build_cnn
from config.settings import DATASET_DIR, MODELS_DIR

IMAGE_SIZE = (128, 128)
EPOCHS = 60
BATCH_SIZE = 16  # increased from 8 — more stable BatchNorm statistics

MODEL_PATH = os.path.join(MODELS_DIR, "face_cnn.keras")  # .keras, not legacy .h5
LABELS_PATH = os.path.join(MODELS_DIR, "class_labels.json")


def load_dataset():
    images = []
    labels = []
    class_names = sorted(os.listdir(DATASET_DIR))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(DATASET_DIR, c))]

    for label_index, student_name in enumerate(class_names):
        folder = os.path.join(DATASET_DIR, student_name)
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label_index)

    return np.array(images), np.array(labels), class_names


def main():
    print("Loading dataset...")
    X, y, class_names = load_dataset()

    if len(X) == 0:
        print("No images found in dataset/. Aborting.")
        return

    # Sanity check — confirms roughly balanced classes, catches loading bugs early
    unique, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(X)} images across {len(class_names)} classes: {class_names}")
    print(f"Per-class counts: {dict(zip([class_names[i] for i in unique], counts))}")

    X = X.astype("float32") / 255.0
    y_categorical = to_categorical(y, num_classes=len(class_names))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
    )

    model = build_cnn(input_shape=(*IMAGE_SIZE, 3), num_classes=len(class_names))
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f)

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class labels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    main()