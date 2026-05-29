"""
  ocr.py

  Tesseract OCR module for extracting text and bounding box coordinates
  from preprocessed document images.

  LayoutLMv3 requires three inputs per word: the word itself, its position
  on the page (bounding box), and the visual region around it. This module
  provides the first two — text and layout positions.

  Bounding boxes are normalized to [0, 1000] range as required by LayoutLMv3.
  """

import pytesseract
import numpy as np
from PIL import Image
from dataclasses import dataclass

@dataclass
class OCRResult:
    """
    Holds the structured output from a single OCR pass on one document page.
    Attributes:
        words: List of extracted words in reading order.
        boxes: List of bounding boxes — one per word.
               Each box is [x_min, y_min, x_max, y_max] normalized to [0, 1000].
        raw_text: Full extracted text as a single string (for display/search).
        confidence: Average OCR confidence score across all words (0-100).
        word_confidences: Per-word confidence scores for filtering low-quality words.
    """
    words: list[str]
    boxes: list[list[int]]
    raw_text: str                                                                                                                                                                                                                         
    confidence: float
    word_confidences: list[float]
def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """
    Normalizes a bounding box from pixel coordinates to [0, 1000] range.
    LayoutLMv3 expects all bounding boxes normalized to a 1000x1000 grid
    regardless of the actual image size. This makes the model's position
    embeddings consistent across documents of different resolutions.
    Args:
        box: [x_min, y_min, x_max, y_max] in pixel coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
    Returns:
        Normalized box [x_min, y_min, x_max, y_max] in [0, 1000] range.
    """
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]
def run_ocr(image: np.ndarray, min_confidence: float = 30.0) -> OCRResult:
    """
    Runs Tesseract OCR on a preprocessed document image.
    Extracts words and their bounding boxes. Filters out low-confidence
    detections (empty strings, punctuation-only tokens, and words below
    the confidence threshold) to keep the output clean for LayoutLMv3.
    Args:
        image: Preprocessed numpy array from image_processor.preprocess().
               Must be grayscale or binary for best results.
        min_confidence: Minimum Tesseract confidence score (0-100) to keep
                        a word. Words below this are dropped. Default 30.0.
    Returns:
        OCRResult containing words, normalized boxes, raw text, and confidence scores.
    """
    # Convert numpy array to PIL for pytesseract
    pil_image = Image.fromarray(image)
    height, width = image.shape[:2]
    # Get detailed OCR data including bounding boxes and confidence scores
    ocr_data = pytesseract.image_to_data(
        pil_image,
        output_type=pytesseract.Output.DICT,
        config="--psm 3"  # psm 3 = fully automatic page segmentation
    )
    words = []
    boxes = []
    confidences = []
    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        word = ocr_data["text"][i].strip()
        conf = float(ocr_data["conf"][i])
        # Skip empty strings, low-confidence detections, and whitespace
        if not word or conf < min_confidence:
            continue

        # Build pixel bounding box from Tesseract output
        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]
        pixel_box = [x, y, x + w, y + h]
        # Normalize to [0, 1000] for LayoutLMv3
        normalized = normalize_box(pixel_box, width, height)
        words.append(word)
        boxes.append(normalized)
        confidences.append(conf)
    raw_text = pytesseract.image_to_string(pil_image, config="--psm 3")
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    return OCRResult(
        words=words,
        boxes=boxes,
        raw_text=raw_text.strip(),
        confidence=avg_confidence,
        word_confidences=confidences
    )
def compute_cer(predicted_text: str, ground_truth_text: str) -> float:
    """
    Computes Character Error Rate (CER) between OCR output and ground truth.
    CER measures OCR quality at the character level — more granular than
    word error rate. A CER of 0.0 means perfect OCR. A CER of 0.1 means
    10% of characters were wrong.
    Used in the evaluation module to measure how much OCR errors
    contribute to overall model accuracy loss.
    Formula: CER = (substitutions + insertions + deletions) / len(ground_truth)
    Args:
        predicted_text: Text extracted by Tesseract.
        ground_truth_text: The correct text (from dataset annotations).
    Returns:
        CER as a float between 0.0 (perfect) and 1.0+ (poor).
    """
    if not ground_truth_text:
        return 0.0
    # Levenshtein distance at character level
    pred = list(predicted_text)
    truth = list(ground_truth_text)
    dp = [[0] * (len(truth) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(truth) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(truth) + 1):
            if pred[i - 1] == truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(pred)][len(truth)] / len(truth)



'''

What each part does in plain terms:
┌───────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │     Part      │                                              What it does                                               │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ OCRResult     │ A clean data container — instead of returning 4 separate lists, we return one object with named fields  │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ normalize_box │ Converts pixel coordinates to [0, 1000] — LayoutLMv3 always expects this scale regardless of image size │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ run_ocr       │ The main function — runs Tesseract, filters garbage words, returns clean words + positions              │
  ├───────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ compute_cer   │ Measures OCR accuracy — used in Step 8 (evaluation) to quantify how much OCR errors hurt the model      │
  └───────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Why min_confidence=30: Tesseract gives every word a confidence score 0-100. Words below 30 are usually noise, partial characters, or scan artifacts — keeping them would confuse LayoutLMv3.

  Why --psm 3: Page Segmentation Mode 3 tells Tesseract to automatically detect columns, paragraphs, and reading order — the right setting for general documents.


'''