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


def segment_and_normalize_leaf(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
  """Preprocess a leaf image aligned with Islam et al. (2019).

  Steps:
  1. Convert BGR to Grayscale.
  2. Segment foreground leaf from white background using Otsu thresholding.
  3. Calculate central moments to find major axis orientation angle and centroid.
  4. Perform affine transformation to align major axis vertically and center centroid.
  5. Crop leaf bounding box and resize to fixed frame dimensions (width, height).
  """
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

  # Invert so leaf foreground is positive (255) on white background (0)
  _, thresh = cv2.threshold(
      gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

  leaf_cnt = max(contours, key=cv2.contourArea)
  if cv2.contourArea(leaf_cnt) < 100:
    return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

  moments = cv2.moments(leaf_cnt)
  if moments["m00"] == 0:
    return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

  cx = moments["m10"] / moments["m00"]
  cy = moments["m01"] / moments["m00"]

  mu20 = moments["mu20"] / moments["m00"]
  mu02 = moments["mu02"] / moments["m00"]
  mu11 = moments["mu11"] / moments["m00"]

  # Compute principal axis angle
  theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
  angle_deg = np.degrees(theta)

  # Rotate so the major axis is vertically aligned
  rot_deg = angle_deg - 90.0 if abs(angle_deg) > 45 else angle_deg

  h, w = gray.shape
  rot_mat = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
  # Translate centroid to image center
  rot_mat[0, 2] += w / 2.0 - cx
  rot_mat[1, 2] += h / 2.0 - cy

  aligned_gray = cv2.warpAffine(gray, rot_mat, (w, h), borderValue=255)
  aligned_thresh = cv2.warpAffine(thresh, rot_mat, (w, h), borderValue=0)

  aligned_contours, _ = cv2.findContours(
      aligned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if aligned_contours:
    aligned_cnt = max(aligned_contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(aligned_cnt)
    if bw > 10 and bh > 10:
      cropped = aligned_gray[by : by + bh, bx : bx + bw]
      return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

  return cv2.resize(aligned_gray, target_size, interpolation=cv2.INTER_AREA)


def to_gray_resize(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
  """Standard entry point for leaf preprocessing."""
  return segment_and_normalize_leaf(image_bgr, target_size=target_size)


def preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
  """Return original RGB image and preprocessed normalized grayscale image."""
  image_bgr = load_bgr(image_path)
  image_gray = segment_and_normalize_leaf(image_bgr, target_size=target_size)

  return image_bgr, image_gray


