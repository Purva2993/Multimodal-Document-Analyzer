"""
  pipeline.py
                                                                                                                                                                                                                                            
  End-to-end inference pipeline for document question answering.

  Orchestrates the full flow from raw document input to final answer:
      Raw document (PDF or image)
          → image_processor.py  (clean the image)
          → ocr.py              (extract text + bounding boxes)
          → model.py            (answer the question)
        → groundedness check  (verify answer is in the document)
This module is the single entry point used by the FastAPI backend.
All other modules are implementation details — the API only calls pipeline.py.
"""
import fitz
import torch
from PIL import Image
from pathlib import Path
from src.config import get_settings
from src.image_processor import preprocess_pil
from src.ocr import run_ocr
from src.model import load_model, load_processor, predict, is_grounded
settings = get_settings()
# Load model and processor once at startup — not on every request.
# These are heavy objects (~500MB). Loading per request would make
# the API unusably slow (10-30 seconds per call).
_processor = None
_model = None
def get_model_and_processor():                                                                                                                                                                                                            
    """
    Returns the shared model and processor instances, loading them on first call.
    Uses module-level singletons so the model is loaded once when the
    application starts, then reused for every subsequent request.
    Returns:
        Tuple of (LayoutLMv3ForQuestionAnswering, LayoutLMv3Processor)
    """
    global _processor, _model
    if _processor is None or _model is None:
        print("Loading LayoutLMv3 model and processor...")
        _processor = load_processor()
        _model = load_model()
        print("Model ready.")
    return _model, _processor
def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """
    Converts each page of a PDF into a PIL image.
    Tesseract and LayoutLMv3 work on images, not PDFs directly.
    PyMuPDF (fitz) renders each PDF page at 300 DPI — the minimum
    resolution for reliable OCR.
    Args:
        pdf_path: Path to the PDF file.
    Returns:
        List of PIL Images, one per page, in page order.
    Raises:
        FileNotFoundError: If the PDF path does not exist.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        # mat scales the page — 300/72 = ~4.17x gives us 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(pil_image)
    doc.close()
    return images

def process_image(image: Image.Image) -> tuple:
    """
    Runs the preprocessing and OCR pipeline on a single document image.
    Applies the full OpenCV preprocessing pipeline to clean the image,
    then runs Tesseract to extract words and bounding boxes.
    Args:
        image: PIL Image of a document page (raw, unprocessed).
    Returns:
        Tuple of:
            - words: List of extracted words in reading order
            - boxes: Normalized bounding boxes [x_min, y_min, x_max, y_max]
            - raw_text: Full document text as a string
            - ocr_confidence: Average OCR confidence score (0-100)
            - dpi_warning: True if image resolution may be too low
    """
    preprocessed = preprocess_pil(image)
    ocr_result = run_ocr(preprocessed)
    return (
        ocr_result.words,
        ocr_result.boxes,
        ocr_result.raw_text,
        ocr_result.confidence,
        False
    )
def answer_question(
    document_path: str,
    question: str,
    page_number: int = 0,
) -> dict:
    """
    Full pipeline: takes a document and question, returns the answer.
    This is the primary function called by the FastAPI backend.
    Handles both PDF and image files. For multi-page PDFs, answers
    from the specified page (default: first page).
    Args:
        document_path: Path to a PDF or image file (PNG, JPG, TIFF).
        question: Natural language question about the document.
        page_number: Page index to query for PDFs (0-indexed). Default 0.
    Returns:
        dict with keys:
            - answer: Extracted answer string
            - confidence: Model confidence score (0.0 to 1.0)
            - is_reliable: True if confidence >= threshold (default 0.5)
            - is_grounded: True if answer exists verbatim in document text
            - ocr_confidence: Tesseract confidence for this page (0-100)
            - dpi_warning: True if image resolution may be too low
            - raw_text: Full OCR text of the page (for display/debug)
            - page: Page number that was queried
    """
    path = Path(document_path)
    model, processor = get_model_and_processor()

    # Load document as image
    if path.suffix.lower() == ".pdf":
        images = pdf_to_images(document_path)
        if page_number >= len(images):
            page_number = 0
        image = images[page_number]
    else:
        image = Image.open(document_path).convert("RGB")
    # Run preprocessing + OCR
    words, boxes, raw_text, ocr_confidence, dpi_warning = process_image(image)
    # Handle completely empty OCR output
    if not words:
        return {
            "answer": "Could not extract text from this document.",
            "confidence": 0.0,
            "is_reliable": False,
            "is_grounded": False,
            "ocr_confidence": ocr_confidence,
            "dpi_warning": True,
            "raw_text": "",
            "page": page_number,
        }
    # Run LayoutLMv3 inference
    result = predict(model, processor, image, words, boxes, question)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "is_reliable": result["is_reliable"],
        "is_grounded": is_grounded(result["answer"], raw_text),
        "ocr_confidence": round(ocr_confidence, 2),
        "dpi_warning": dpi_warning,
        "raw_text": raw_text,
        "page": page_number,
    }

def answer_question_from_image(
    image: Image.Image,
    question: str,
) -> dict:
    """
    Runs the full pipeline on an already-loaded PIL image.
    Used by the Gradio frontend which receives uploaded images directly
    as PIL objects — avoids saving to disk and reloading.
    Args:
        image: PIL Image of the document (already loaded).
        question: Natural language question about the document.
    Returns:
        Same dict structure as answer_question().
    """
    model, processor = get_model_and_processor()
    words, boxes, raw_text, ocr_confidence, dpi_warning = process_image(image)
    if not words:
        return {
            "answer": "Could not extract text from this document.",
            "confidence": 0.0,
            "is_reliable": False,
            "is_grounded": False,
            "ocr_confidence": 0.0,
            "dpi_warning": True,
            "raw_text": "",
            "page": 0,
        }

    result = predict(model, processor, image, words, boxes, question)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "is_reliable": result["is_reliable"],
        "is_grounded": is_grounded(result["answer"], raw_text),
        "ocr_confidence": round(ocr_confidence, 2),
        "dpi_warning": dpi_warning,
        "raw_text": raw_text,
        "page": 0,
    }

'''

  What each part does in plain terms:

  ┌────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
  │            Part            │                                        What it does                                        │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ get_model_and_processor    │ Loads the model once on startup, reuses it forever — prevents 30-second delays per request │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ pdf_to_images              │ Converts PDF pages to images at 300 DPI — Tesseract needs images, not PDFs                 │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ process_image              │ Runs preprocessing + OCR on one page — returns words, boxes, and text                      │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ answer_question            │ Main function — takes file path + question, returns complete answer dict                   │
  ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ answer_question_from_image │ Same but accepts PIL image directly — used by Gradio to avoid disk I/O                     │
  └────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘

  Why module-level singletons for model + processor: LayoutLMv3 is ~500MB. Loading it on every API request would make the app take 10-30 seconds per question. Loading once at startup and reusing is standard practice for ML serving.

'''