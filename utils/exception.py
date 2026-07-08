class CameraError(Exception):
    """Raised when the camera fails to open or read a frame."""
    pass


class NoFaceDetectedError(Exception):
    """Raised when no face is found in an image where one was expected."""
    pass


class LowImageQualityError(Exception):
    """Raised when an image fails blur/brightness/size checks."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class DuplicateStudentError(Exception):
    """Raised when a face matches an already-registered student."""
    pass