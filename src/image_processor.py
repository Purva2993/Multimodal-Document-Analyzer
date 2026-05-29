"""
  image_processor.py

  OpenCV-based image preprocessing pipeline for document images.
  Runs before OCR to normalize and clean document images.

  Poor image quality is the single biggest cause of OCR errors.
  This pipeline corrects the most common issues found in scanned
  documents: tilt, noise, low contrast, and wrong color mode.

  Pipeline order:
      load → grayscale → DPI check → deskew → denoise → binarize → enhance contrast
  """

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk into a numpy array for OpenCV processing.
    Args:
        image_path: Path to the image file (PNG, JPG, TIFF supported).
    Returns:
        Image as a numpy array in BGR format (OpenCV default).
    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file cannot be read as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to grayscale.
    Grayscale removes color information that adds noise without helping
    OCR. Tesseract reads text as dark pixels on a light background —
    color is irrelevant and slows processing.
    Args:
        image: BGR numpy array (OpenCV format).
    Returns:
        Single-channel grayscale numpy array.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def check_dpi(image: np.ndarray, min_dpi: int = 300) -> dict:
    """
    Estimates image resolution and warns if below the minimum for OCR.
    Tesseract requires at least 300 DPI to reliably read text.
    Below 150 DPI, OCR accuracy drops significantly regardless
    of any other preprocessing applied.
    Note: DPI metadata is often missing from scanned images.
    This function estimates resolution from image dimensions as a proxy.
    Args:
        image: Grayscale numpy array.
        min_dpi: Minimum acceptable DPI. Default is 300.
    Returns:
        dict with keys:
            - width: image width in pixels
            - height: image height in pixels
            - warning: True if resolution may be too low for reliable OCR
    """
    height, width = image.shape[:2]
    # A standard A4 page at 300 DPI is 2480 x 3508 pixels
    estimated_low_res = width < 1000 or height < 1000
    return {
        "width": width,
        "height": height,
        "warning": estimated_low_res,
        "message": (
            f"Image may be too low resolution ({width}x{height}px). "
            f"For reliable OCR, use documents scanned at {min_dpi}+ DPI."
        ) if estimated_low_res else "Resolution looks good."
    }
def deskew(image: np.ndarray) -> np.ndarray:
    """
    Straightens a tilted or rotated document image.
    Even a 2-degree tilt significantly reduces OCR accuracy because
    Tesseract reads text in horizontal lines. This function detects
    the skew angle using image moments on detected contours and
    rotates the image to correct it.
    Args:
        image: Grayscale numpy array.
    Returns:
        Deskewed grayscale numpy array, same dimensions as input.
    """
    # Threshold to find text regions
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find coordinates of all non-zero (text) pixels
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image

    # Fit a rotated bounding box around all text pixels
    angle = cv2.minAreaRect(coords)[-1]
    # Correct the angle — minAreaRect returns angles in [-90, 0)
    if angle < -45:
        angle = 90 + angle

    # Rotate image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed

def denoise(image: np.ndarray) -> np.ndarray:
    """
    Removes noise, speckles, and scan artifacts from the image.
    Scanners introduce random pixel noise (speckles) that Tesseract
    can misread as punctuation or characters. Non-local means denoising
    preserves text edges while smoothing out background noise.
    Args:
        image: Grayscale numpy array.

    Returns:
        Denoised grayscale numpy array.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
def binarize(image: np.ndarray) -> np.ndarray:
    """
    Converts image to pure black and white using adaptive thresholding.
    Binarization makes text sharply black on a white background.
    Adaptive thresholding (vs global) handles uneven lighting across
    the page — common in photos of documents or older scans where
    one corner is darker than the other.
    Args:
        image: Grayscale numpy array.
    Returns:
        Binary (black and white) numpy array.
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhances contrast to make faint or faded text more readable.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    which boosts local contrast rather than global — avoids
    over-brightening already clear regions while lifting faded text.
    Args:
        image: Grayscale numpy array.
    Returns:
        Contrast-enhanced grayscale numpy array.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
def preprocess(image_path: str) -> tuple[np.ndarray, dict]:
    """
    Runs the full preprocessing pipeline on a document image.
    This is the main entry point called by the OCR module.
    Applies all steps in the correct order: grayscale → DPI check
    → deskew → denoise → binarize → enhance contrast.
    Args:
        image_path: Path to the raw document image.
    Returns:
        Tuple of:
            - Preprocessed image as numpy array, ready for Tesseract
            - DPI check result dict (contains warning if resolution is low)
    """
    image = load_image(image_path)
    image = to_grayscale(image)
    dpi_info = check_dpi(image)
    image = deskew(image)
    image = denoise(image)
    image = enhance_contrast(image)
    image = binarize(image)
    return image, dpi_info
def preprocess_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Runs the preprocessing pipeline on a PIL image (used for PDF pages).
    PyMuPDF returns PIL images when converting PDF pages. This function
    accepts PIL format directly so the PDF processing module doesn't
    need to save intermediate files to disk.
    Args:
        pil_image: PIL Image object (from PDF page conversion).
    Returns:
        Preprocessed image as numpy array, ready for Tesseract.
    """
    image = np.array(pil_image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = to_grayscale(image)
    image = deskew(image)
    image = denoise(image)
    image = enhance_contrast(image)
    image = binarize(image)
    return image
  
'''

  What each function does in plain terms:

  ┌──────────────────┬───────────────────────────────────────────────────────────────────────────────┐
  │     Function     │                                 What it fixes                                 │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ to_grayscale     │ Removes color — OCR doesn't need it, it just adds noise                       │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ check_dpi        │ Warns you before OCR runs if the image is too low resolution to read reliably │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ deskew           │ Straightens a tilted scan — a 2-degree tilt breaks OCR line detection         │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ denoise          │ Removes scanner speckles that get misread as punctuation                      │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ binarize         │ Pure black/white — sharpens text edges, handles uneven lighting               │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ enhance_contrast │ Lifts faded or light text so Tesseract can see it                             │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ preprocess       │ The one function everyone else calls — runs all steps in order                │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ preprocess_pil   │ Same pipeline but accepts PDF pages directly from PyMuPDF                     │
  └──────────────────┴───────────────────────────────────────────────────────────────────────────────┘

'''