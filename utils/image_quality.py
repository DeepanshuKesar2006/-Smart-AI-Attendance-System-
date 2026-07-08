import cv2
import numpy as np

MIN_BLUR_THRESHOLD = 60.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
MIN_FACE_SIZE = 80


def is_blurry(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < MIN_BLUR_THRESHOLD


def is_bad_brightness(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS


def is_face_too_small(box):
    _, _, w, h = box
    return w < MIN_FACE_SIZE or h < MIN_FACE_SIZE


def check_quality(face_image, box):
    """Returns (is_good, reason)."""
    if is_face_too_small(box):
        return False, "Face too small"
    if is_blurry(face_image):
        return False, "Image too blurry"
    if is_bad_brightness(face_image):
        return False, "Poor lighting"
    return True, "OK"