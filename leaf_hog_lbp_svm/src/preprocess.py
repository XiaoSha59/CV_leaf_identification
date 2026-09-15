from pathlib import Path

import cv2
import numpy as np 

from src.config import IMAGE_SIZE

def load_bgr(image_path: str | Path) -> np.ndarray:
    """Load an image as a BGR uint8 array."""
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image_bgr

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for matplotlib visualization."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def to_gray_resize(image_bgr: np.ndarray, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Convert to grayscale and resize image"""
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.resize(
        image_gray, 
        target_size,
        interpolation = cv2.INTER_AREA,
    )

def preprocess_image(image_path: str | Path, target_size: tuple[int,int] = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return original image and preprocessed grayscale image"""
    image_bgr = load_bgr(image_path)
    image_gray = to_gray_resize(image_bgr, target_size = target_size)

    return image_bgr, image_gray

