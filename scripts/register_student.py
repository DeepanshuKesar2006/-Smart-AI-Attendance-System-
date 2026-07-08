import sys
import os
import cv2
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.camera import Camera
from app.detector import FaceDetector
from app.embedding import EmbeddingGenerator
from app.registration import RegistrationManager
from utils.image_quality import check_quality
from config.settings import FONT, GREEN, RED, DATASET_DIR

TARGET_IMAGE_COUNT = 20


def main():
    student_name = input("Enter new student name: ").strip()
    if not student_name:
        print("Name cannot be empty.")
        return

    save_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(save_dir, exist_ok=True)

    camera = Camera()
    detector = FaceDetector()
    embedder = EmbeddingGenerator()
    manager = RegistrationManager()

    captured = 0
    print("Look at the camera. Press 'q' to quit early.")

    while captured < TARGET_IMAGE_COUNT:
        success, frame = camera.read()
        if not success:
            break

        display = frame.copy()
        faces = detector.detect_faces(frame)

        for box in faces:
            x, y, w, h = box
            face_crop = detector.crop_face(frame, box)
            good, reason = check_quality(face_crop, box)

            color = GREEN if good else RED
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, reason, (x, y - 10), FONT, 0.6, color, 2)

            if good:
                embedding = embedder.get_embedding(face_crop)
                is_dup, matched_name = manager.is_duplicate(embedding, exclude_name=student_name)

                if is_dup:
                    print(f"This face matches an existing student: {matched_name}. Registration stopped.")
                    camera.release()
                    cv2.destroyAllWindows()
                    return

                image_path = os.path.join(save_dir, f"{student_name}_{captured}.jpg")
                cv2.imwrite(image_path, face_crop)
                captured += 1
                time.sleep(0.3)

        cv2.putText(display, f"Captured: {captured}/{TARGET_IMAGE_COUNT}", (20, 30), FONT, 0.7, GREEN, 2)
        cv2.imshow("Register Student", display)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if captured > 0:
        manager.register_from_folder(student_name, save_dir)
        print(f"Registered {student_name} with {captured} images.")
    else:
        print("No images captured. Registration cancelled.")


if __name__ == "__main__":
    main()