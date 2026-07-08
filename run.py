import cv2
import time

from app.camera import Camera
# from app.recognizer import FaceRecognizer
from app.cnn_recognizer import CNNFaceRecognizer as FaceRecognizer
from app.attendance import AttendanceManager
from config.settings import FONT, GREEN, RED, BLUE


def main():
    camera = Camera()
    recognizer = FaceRecognizer()
    attendance = AttendanceManager()

    previous_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        display = frame.copy()
        faces = recognizer.detect_faces(frame)

        for box in faces:
            x, y, w, h = box

            try:
                name, confidence = recognizer.recognize(frame, box)
            except Exception:
                continue

            color = RED if name == "Unknown" else GREEN
            status_text = ""

            if name != "Unknown":
                marked, message = attendance.mark_attendance(name)
                if marked:
                    status_text = "Marked!"
                    color = (0, 255, 255)  # highlight the moment it's marked

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, name, (x, y - 35), FONT, 0.8, color, 2)
            cv2.putText(display, f"{confidence:.1f}% {status_text}", (x, y - 10), FONT, 0.7, BLUE, 2)

        current = time.time()
        fps = 1 / (current - previous_time)
        previous_time = current
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 30), FONT, 0.7, BLUE, 2)

        cv2.imshow("Smart AI Attendance System", display)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()